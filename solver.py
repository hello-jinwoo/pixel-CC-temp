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
import cv2

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
		self.aux_coeff = config.aux_coeff
		self.output_type = config.output_type
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
		self.white_level = config.white_level
		self.log_interval = config.log_interval

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
			mae_abstract_list = torch.Tensor([])
			mae_full_list = torch.Tensor([])
			psnr_list = torch.Tensor([])

			# Train
			self.net.train()
			for i, batch in enumerate(self.train_loader):

				input_rgb = batch['input_rgb'].to(self.device) # [B, 3, H, W]
				input_tensor = batch["input"].to(self.device) # [B, 3, H, W]
				gt_illum_tensor = batch['illum_gt'].to(self.device) # [B, 2, H, W]
				gt_image_tensor = batch['gt'].to(self.device) # [B, 2, H, W] or # [B, 3, H, W]
				gt_image_rgb_tensor = batch['gt_rgb'].to(self.device) # [B, 3, H, W]
				abstract_gt_illum = utils.get_abstract_illum_map(gt_illum_tensor, self.abstract_pool) # [B, 2, H // P, W // P]
				gt_illum_tensor_with_g = torch.ones_like(input_rgb) # [B, 3, H, W]
				gt_illum_tensor_with_g[:, [0,2], ...] = gt_illum_tensor

				output_tensor, abstract_output_tensor = self.net(input_tensor)
				output_rgb = torch.clip(utils.apply_wb(input_rgb, output_tensor, self.output_type), 0, self.white_level)
				if self.output_type == 'uv':
					output_illum =  input_rgb / (output_rgb + 1e-8)
					gt_illum = input_rgb / (gt_image_rgb_tensor + 1e-8)
				elif self.output_type == 'rgb':
					output_illum = output_tensor

				loss_abstract_illum = self.criterion(abstract_output_tensor.float(), abstract_gt_illum.float())
				loss_full_image = self.criterion(output_tensor.float(),gt_image_tensor.float())
				loss = loss_abstract_illum * self.aux_coeff + loss_full_image * (1 - self.aux_coeff)

				self.net.zero_grad()
				loss.backward()
				self.optimizer.step()

				# mae_full_per_batch, _, _ = metrics.get_mae(output_illum, gt_illum_tensor_with_g)
				mae_full_per_batch, _, _ = metrics.get_mae(output_illum, gt_illum)
				mae_full_list = torch.cat([mae_full_list, mae_full_per_batch.detach().cpu()])
				mae_abstract_per_batch, _, _ = metrics.get_mae(abstract_output_tensor, abstract_gt_illum, include_g=False)
				mae_abstract_list = torch.cat([mae_abstract_list, mae_abstract_per_batch.detach().cpu()])
				psnr_per_batch = metrics.get_psnr(output_rgb.permute(0,2,3,1).detach().cpu().numpy(),
												  gt_image_rgb_tensor.permute(0,2,3,1).detach().cpu().numpy(),
												  self.white_level)
				psnr_list = torch.cat([psnr_list, torch.Tensor([psnr_per_batch])])

				# print training log & tensorboard logging (every iteration)
				if i % self.log_interval == 0:
					mae_abstract = torch.mean(mae_abstract_list)
					mae_full = torch.mean(mae_full_list)
					psnr = torch.mean(psnr_list)
					print(f'[Train] Epoch [{epoch+1} / {self.num_epochs}] | ' \
						  f'Batch [{i+1} / {len(self.train_loader)}] | ' \
						  f'Loss: {loss.item():.5f} | ' \
						  f'Loss_abstract: {loss_abstract_illum.item():.5f} | ' \
						  f'Loss_full: {loss_full_image.item():.5f} | ' \
						  f'MAE_abstract: {mae_abstract:.3f} | ' \
						  f'MAE_full: {mae_full:.3f} | ' \
						  f'PSNR: {psnr:.2f}')
					self.writer.add_scalar('train/Loss', loss.item(), epoch*len(self.train_loader)+i)
					self.writer.add_scalar('train/Loss_full_image', loss_full_image.item(), epoch*len(self.train_loader)+i)
					self.writer.add_scalar('train/loss_abstract_illum', loss_abstract_illum.item(), epoch*len(self.train_loader)+i)
					self.writer.add_scalar('train/MAE_abstract', mae_abstract, epoch*len(self.train_loader)+i)
					self.writer.add_scalar('train/MAE_full', mae_full, epoch*len(self.train_loader)+i)
					self.writer.add_scalar('train/PSNR', psnr, epoch*len(self.train_loader)+i)

				mae_abstract_list = torch.Tensor([])
				mae_full_list = torch.Tensor([])
				psnr_list = torch.Tensor([])

			# Validation
			val_score_abstract_illum = 0
			val_score_full_image = 0
			val_score = 0
			n_val = 0
			self.net.eval()
			for i, batch in enumerate(self.val_loader):
				input_rgb = batch['input_rgb'].to(self.device) # [B, 3, H, W]
				input_tensor = batch["input"].to(self.device) # [B, 3, H, W]
				gt_illum_tensor = batch['illum_gt'].to(self.device) # [B, 2, H, W]
				gt_image_tensor = batch['gt'].to(self.device) # # [B, 2, H, W] or [B, 3, H, W]
				gt_image_rgb_tensor = batch['gt_rgb'].to(self.device) # [B, 3, H, W]
				abstract_gt_illum = utils.get_abstract_illum_map(gt_illum_tensor, self.abstract_pool)
				gt_illum_tensor_with_g = torch.ones_like(input_rgb) # [B, 3, H, W]
				gt_illum_tensor_with_g[:, [0,2], ...] = gt_illum_tensor

				output_tensor, abstract_output_tensor = self.net(input_tensor)
				output_rgb = torch.clip(utils.apply_wb(input_rgb, output_tensor, self.output_type), 0, self.white_level)
				if self.output_type == 'uv':
					output_illum =  input_rgb / (output_rgb + 1e-8)
					gt_illum = input_rgb / (gt_image_rgb_tensor + 1e-8)
				elif self.output_type == 'rgb':
					output_illum = output_tensor

				loss_abstract_illum = self.criterion(abstract_output_tensor.float(), abstract_gt_illum.float())
				loss_full_image = self.criterion(output_tensor.float(),gt_image_tensor.float())
				loss = float(loss_abstract_illum * self.aux_coeff + loss_full_image * (1 - self.aux_coeff))

				minibatch_size = len(input_tensor)
				n_val += minibatch_size
				val_score_abstract_illum += float(loss_abstract_illum * minibatch_size)
				val_score_full_image += float(loss_full_image * minibatch_size)
				
				# mae_full_per_batch, _, _ = metrics.get_mae(output_illum, gt_illum_tensor_with_g)
				mae_full_per_batch, _, _ = metrics.get_mae(output_illum, gt_illum)
				mae_full_list = torch.cat([mae_full_list, mae_full_per_batch.detach().cpu()])
				mae_abstract_per_batch, _, _ = metrics.get_mae(abstract_output_tensor, abstract_gt_illum, include_g=False)
				mae_abstract_list = torch.cat([mae_abstract_list, mae_abstract_per_batch.detach().cpu()])
				psnr_per_batch = metrics.get_psnr(output_rgb.permute(0,2,3,1).detach().cpu().numpy(),
												  gt_image_rgb_tensor.permute(0,2,3,1).detach().cpu().numpy(),
												  self.white_level)
				psnr_list = torch.cat([psnr_list, torch.Tensor([psnr_per_batch])])

			val_score_abstract_illum /= n_val
			val_score_full_image /= n_val
			val_score = val_score_abstract_illum * self.aux_coeff + val_score_full_image * (1 - self.aux_coeff)

			mae_abstract = torch.mean(mae_abstract_list)
			mae_full = torch.mean(mae_full_list)

			# print validation log & tensorboard logging (once per epoch)
			print(f'[Valid] Epoch [{epoch+1} / {self.num_epochs}] | ' \
				  f'Loss: {val_score:.5f} | ' \
				  f'Loss_abstract: {val_score_abstract_illum:.5f} | ' \
				  f'Loss_full: {val_score_full_image:.5f} | ' \
				  f'MAE_abstract: {mae_abstract:.3f} | ' \
				  f'MAE_full: {mae_full:.3f} | ' \
				  f'PSNR: {psnr:.2f}')
			self.writer.add_scalar('validation/Loss', val_score, epoch)
			self.writer.add_scalar('validation/loss_abstract_illum', val_score_abstract_illum, epoch)
			self.writer.add_scalar('validation/Loss_full_image', val_score_full_image, epoch)
			self.writer.add_scalar('validation/MAE_abstract', mae_abstract, epoch)
			self.writer.add_scalar('validation/MAE_full', mae_full, epoch)
			self.writer.add_scalar('validation/PSNR', psnr, epoch)

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
			loss = float(loss_abstract_illum * self.aux_coeff + loss_full_image * (1 - self.aux_coeff))
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