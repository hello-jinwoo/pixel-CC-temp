import torch,os,time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from model import U_Net as U_Net
import metrics
import utils
import sys

class Solver():
	def __init__(self, config, train_loader, val_loader, test_loader):
		# Misc
		self.start_time = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Training config
		self.mode = config.mode
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size
		self.lr = config.lr
		self.criterion = nn.MSELoss(reduction='mean')

		# Dataloader
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader

		# Path
		self.checkpoint = config.checkpoint

		if self.mode == 'train':
			self.model_path = os.path.join(config.model_root,self.start_time)
			self.result_path = os.path.join(config.result_root,self.start_time+'_train')
			self.log_path = os.path.join(config.log_root,self.start_time)
			self.writer = SummaryWriter(self.log_path)
		elif self.mode == 'test':
			self.model_path = os.path.join(config.model_root,self.checkpoint)
			self.result_path = os.path.join(config.result_root,self.checkpoint+'_test')

		if os.path.isdir(self.model_path) == False and self.mode == 'train':	
			os.makedirs(self.model_path)
		if os.path.isdir(self.result_path) == False:
			os.makedirs(self.result_path)

		# Misc
		self.save_epoch = config.save_epoch
		self.multi_gpu = config.multi_gpu
		self.abstract_pool = config.abstract_pool

		self.build_model()

	def build_model(self):
		self.net = U_Net()
		# # build resnet18 & modify last fc layer
		# self.net = models.resnet50()
		# self.net.fc = nn.Sequential(
		#     nn.Linear(2048,1024),
		#     nn.Linear(1024,512),
		#     nn.Linear(512,256),
		#     nn.Linear(256,4)
		# )

		if self.mode == 'test':
			# load model from checkpoint
			ckpt = os.path.join(self.model_path,'best.pt')
			print("[Model]\tLoad model from checkpoint :", ckpt)
			self.net.load_state_dict(torch.load(ckpt),strict=False)

		# multi-GPU
		if torch.cuda.device_count() > 1 and self.multi_gpu == 1: 
			self.net = nn.DataParallel(self.net)

		# gpu & optimizer
		self.net.to(self.device)
		self.optimizer = torch.optim.Adam(list(self.net.parameters()), self.lr)
		print("[Model]\tBuild complete.")

	def train(self):
		print("[Train]\tStart training process.")
		best_val_score = 987654321.

		for epoch in range(self.num_epochs):
			# Train
			self.net.train()
			mae_list = torch.Tensor([])
			for i, batch in enumerate(self.train_loader):

				input_tensor = batch["input"].to(self.device)
				gt_illum_tensor = batch['illum_gt'].to(self.device)
				gt_image_tensor = batch['gt'].to(self.device)
				abstract_gt_illum = utils.get_abstract_illum_map(gt_illum_tensor, self.abstract_pool)

				output_tensor, abstract_output_tensor = self.net(input_tensor)

				loss_abstract_illum = self.criterion(abstract_output_tensor.float(), abstract_gt_illum.float())
				loss_full_image = self.criterion(output_tensor.float(),gt_image_tensor.float())
				loss = loss_abstract_illum + loss_full_image

				self.net.zero_grad()
				loss.backward()
				self.optimizer.step()
				mae_per_batch, _, _ = metrics.get_mae(output_tensor.float(), gt_image_tensor.float())
				mae_list = torch.cat([mae_list, mae_per_batch.detach().cpu()])

				# print training log & tensorboard logging (every iteration)
				if i % 10 == 0:
					mae = torch.mean(mae_list)
					print(f'[Train] Epoch [{epoch+1} / {self.num_epochs}] | ' \
						  f'Batch [{i+1} / {len(self.train_loader)}] | ' \
						  f'Loss: {loss.item():.6f} | ' \
						  f'MAE: {mae:.4f}')
				self.writer.add_scalar('Loss/train', loss.item(), epoch*len(self.train_loader)+i)
				self.writer.add_scalar('MAE/train', mae, epoch*len(self.train_loader)+i)
				mae_list = torch.Tensor([])
			# Validation
			val_score = 0
			n_val = 0
			mae_list_val = torch.Tensor([])
			self.net.eval()
			for i, batch in enumerate(self.val_loader):
				input_tensor = batch["input"].to(self.device)
				gt_illum_tensor = batch['illum_gt'].to(self.device)
				gt_image_tensor = batch['gt'].to(self.device)
				abstract_gt_illum = utils.get_abstract_illum_map(gt_illum_tensor, self.abstract_pool)

				minibatch_size = len(input_tensor)
				n_val += minibatch_size

				output_tensor, abstract_output_tensor = self.net(input_tensor)

				loss_abstract_illum = self.criterion(abstract_output_tensor.float(), abstract_gt_illum.float())
				loss_full_image = self.criterion(output_tensor.float(),gt_image_tensor.float())
				loss = float(loss_abstract_illum + loss_full_image)

				val_score += loss * minibatch_size
				mae_per_batch, _, _ = metrics.get_mae(output_tensor.float(), gt_image_tensor.float())
				mae_list = torch.cat([mae_list, mae_per_batch.detach().cpu()])
			val_score /= n_val

			mae = torch.mean(mae_list)
			# print validation log & tensorboard logging (once per epoch)
			print(f'[Valid] Epoch [{epoch+1} / {self.num_epochs}] | ' \
				  f'Loss: {loss:.6f} | ' \
				  f'MAE: {mae:.4f}')
			self.writer.add_scalar('Loss/validation', val_score, epoch)
			self.writer.add_scalar('MAE/train', mae, epoch*len(self.train_loader)+i)

			# Save best model
			if val_score < best_val_score:
				best_val_score = val_score
				if self.multi_gpu == 1:
					best_net = self.net.module.state_dict()
				else:
					best_net = self.net.state_dict()
				torch.save(best_net, os.path.join(self.model_path, 'best.pt'))
				print(f'Best validation score : {best_val_score:.6f}')
			# Save every N epoch
			elif self.save_epoch > 0 and epoch % self.save_epoch == self.save_epoch - 1:
				if self.multi_gpu == 1:
					state_dict = self.net.module.state_dict()
				else:
					state_dict = self.net.state_dict()
				torch.save(best_net, os.path.join(self.model_path, str(epoch)+'.pt'))

	def test(self):
		self.net.eval()

		test_loss = []
		for i, batch in enumerate(self.test_loader):
			place, illum_count, img_id = batch["place"][0], batch["illum_count"][0], batch["img_id"][0]

			input_tensor = batch["input"].to(self.device)
			gt_illum_tensor = batch['illum_gt'].to(self.device)
			gt_image_tensor = batch['gt'].to(self.device)	
			abstract_gt_illum = utils.get_abstract_illum_map(gt_illum_tensor, self.abstract_pool)

			output_tensor, abstract_output_tensor = self.net(input_tensor)
			loss_abstract_illum = self.criterion(abstract_output_tensor.float(), abstract_gt_illum.float())
			loss_full_image = self.criterion(output_tensor.float(),gt_image_tensor.float())
			loss = float(loss_abstract_illum + loss_full_image)
			test_loss.append(loss.item())
			
			# print log
			print(f'[Test] Batch [{i+1} / {len(self.test_loader)}] | ' \
				  f'GT {gt_tensor[0].detach().cpu().numpy()} | ' \
				  f'Pred {output_tensor[0].detach().cpu().numpy()} | ' \
				  f'Loss {loss}')

			# save plot
			output_illum1 = output_tensor[0][0:2].detach().cpu().numpy()
			output_illum2 = output_tensor[0][2:4].detach().cpu().numpy()
			gt_illum1 = gt_tensor[0][0:2].detach().cpu().numpy()
			gt_illum2 = gt_tensor[0][2:4].detach().cpu().numpy()
			plt.plot(output_illum1[0], output_illum1[1], 'ro', label='pred_1')
			plt.plot(output_illum2[0], output_illum2[1], 'r^', label='pred_2')
			plt.plot(gt_illum1[0], gt_illum1[1], 'go', label='gt_1')
			plt.plot(gt_illum2[0], gt_illum2[1], 'g^', label='gt_2')
			plt.axis([0,1,0,1])
			plt.xlabel('R/G')
			plt.ylabel('B/G')
			plt.legend()
			plt.savefig(os.path.join(self.result_path,'_'.join([str(i),place,illum_count,img_id])+'.png'))
			plt.clf()
		
		print(f'Test Loss [Avg] : {np.mean(test_loss):.6f} ' \
			  f'[Min] : {np.min(test_loss):.6f} ' \
			  f'[Med] : {np.median(test_loss):.6f} ' \
			  f'[Max] : {np.max(test_loss):.6f} ')


	def process_illum(self):
		pass

	def plot_labels(self,target):
		if 'train' in target:
			for batch in self.train_loader:
				gt_illum1 = batch["illum1"].numpy()
				gt_illum2 = batch["illum2"].numpy()

				plt.plot(gt_illum1[:,0],gt_illum1[:,2],'ro')
				plt.plot(gt_illum2[:,0],gt_illum2[:,2],'b^')

			plt.xlabel('R/G')
			plt.ylabel('B/G')
			plt.axis([0,4,0,4])
			plt.title('GT Illumination Distribution')
			plt.savefig("gt_train.png")
			plt.clf()
		
		if 'val' in target:
			for batch in self.val_loader:
				gt_illum1 = batch["illum1"].numpy()
				gt_illum2 = batch["illum2"].numpy()

				plt.plot(gt_illum1[:,0],gt_illum1[:,2],'ro')
				plt.plot(gt_illum2[:,0],gt_illum2[:,2],'b^')

			plt.xlabel('R/G')
			plt.ylabel('B/G')
			plt.axis([0,4,0,4])
			plt.title('GT Illumination Distribution')
			plt.savefig("gt_val.png")
			plt.clf()

		if 'test' in target:
			for batch in self.test_loader:
				gt_illum1 = batch["illum1"].numpy()
				gt_illum2 = batch["illum2"].numpy()

				plt.plot(gt_illum1[:,0],gt_illum1[:,2],'ro')
				plt.plot(gt_illum2[:,0],gt_illum2[:,2],'b^')

			plt.xlabel('R/G')
			plt.ylabel('B/G')
			plt.axis([0,4,0,4])
			plt.title('GT Illumination Distribution')
			plt.savefig("gt_test.png")
			plt.clf()