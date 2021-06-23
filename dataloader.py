import os,cv2,json
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

class LSMI(data.Dataset):
    def __init__(self,root,split,image_pool,image_size,
                 input_type='uvl',output_type=None,
                 mask_black=None,mask_highlight=None):
        self.root = root                        # dataset root
        self.split = split                      # 'train' / 'val' / 'test'
        self.image_pool = image_pool            # 1 / 12 / 123
        self.mask_black = mask_black            # None or Masked value for black pixels 
        self.mask_highlight = mask_highlight    # None or Saturation value

        self.image_size = image_size
        self.image_list = sorted([f for f in os.listdir(os.path.join(root,split))
                                 if f.split('_')[-2] in image_pool
                                 and f.endswith(".tiff")
                                 and "gt" not in f])
        
        meta_path = os.path.join(self.root,'meta.json')
        with open(meta_path, 'r') as json_file:
            self.meta_data = json.load(json_file)

        self.input_type = input_type            # uvl / rgb
        self.output_type = output_type          # None / illumination / uv

        print("[Data]\t"+str(self.__len__())+" "+split+" images are loaded from "+root)
        print()

    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information
        input_tensor    : input image (uvl or rgb)
        gt_tensor       : GT (None or illumination or chromaticity)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels
        """

        # parse fname
        fname = self.image_list[idx]
        place, illum_count, img_id = os.path.splitext(fname)[0].split('_')

        # 1. prepare meta information
        ret_dict = {}
        dummy = torch.tensor([-1,-1,-1], dtype=torch.float32)
        ret_dict["illum1"] = torch.tensor(self.meta_data[self.split][place][illum_count+'_'+img_id][0], dtype=torch.float32)
        ret_dict["illum2"], ret_dict["illum3"] = dummy, dummy
        if illum_count == "12":
            ret_dict["illum2"] = torch.tensor(self.meta_data[self.split][place][illum_count+'_'+img_id][1], dtype=torch.float32)
        elif illum_count == "123":
            ret_dict["illum2"] = torch.tensor(self.meta_data[self.split][place][illum_count+'_'+img_id][1], dtype=torch.float32)
            ret_dict["illum3"] = torch.tensor(self.meta_data[self.split][place][illum_count+'_'+img_id][2], dtype=torch.float32)
        ret_dict["fname"] = fname
        ret_dict["place"] = place
        ret_dict["illum_count"] = illum_count
        ret_dict["img_id"] = img_id

        # 2. prepare input
        # load 3-channel rgb tiff image
        input_path = os.path.join(self.root,self.split,fname)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        ret_dict["input_rgb"] = torch.tensor(input_rgb, dtype=torch.float32).permute(2,0,1)

        if self.input_type == 'rgb':
            input_tensor = input_rgb
        elif self.input_type == 'uvl':
            input_tensor = self.rgb2uvl(input_rgb)
        ret_dict["input"] = torch.tensor(input_tensor, dtype=torch.float32).permute(2,0,1)

        # 3. prepare GT
        if self.output_type == 'mixmap':
            raise NotImplementedError

        if self.output_type != None:
            # illum gt
            illum_path = os.path.join(self.root, self.split, os.path.splitext(fname)[0]+'_illum.npy')
            gt_illumination = np.load(illum_path)
            gt_illumination.astype('float32')
            gt_tensor = np.delete(gt_illumination, 1, axis=2)   # delete green channel
            gt_tensor = torch.tensor(gt_tensor, dtype=torch.float32).permute(2,0,1)
            ret_dict["illum_gt"] = gt_tensor
            # image gt
            output_path = os.path.join(self.root, self.split, os.path.splitext(fname)[0]+'_gt.tiff')
            gt_bgr = cv2.imread(output_path, cv2.IMREAD_UNCHANGED).astype('float32')
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
            gt_uvl = self.rgb2uvl(gt_rgb)
            gt_uv = np.delete(gt_uvl, 2, axis=2) # delete l channel
            gt_tensor = gt_uv
            gt_tensor = torch.tensor(gt_tensor, dtype=torch.float32).permute(2,0,1)
            ret_dict["gt"] = gt_tensor
            ret_dict["gt_rgb"] = torch.tensor(gt_rgb).permute(2,0,1)

        # 4. prepare mask
        mask = np.ones_like(input_rgb[:,:,0], dtype='float32')[:,:,None]
        if self.mask_black != None:
            raise NotImplementedError("Implement black pixel masking!")
        if self.mask_highlight != None:
            raise NotImplementedError("Implement highlight masking!")
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
        ret_dict["mask"] = mask

        return ret_dict

    def rgb2uvl(self, img_rgb):
        epsilon = 1e-4
        img_uvl = np.zeros_like(img_rgb, dtype=np.float32)
        img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
        img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
        img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

        return img_uvl

    def __len__(self):
        return len(self.image_list)

def get_loader(config, split):
    dataset = LSMI(root=config.data_root,
                   split=split,
                   image_pool=config.image_pool,
                   image_size=config.image_size,
                   input_type=config.input_type,
                   output_type=config.output_type,
                   mask_black=config.mask_black,
                   mask_highlight=config.mask_highlight)
    
    if split == 'test':
        dataloader = data.DataLoader(dataset,batch_size=1,shuffle=False,
                                     num_workers=config.num_workers)
    elif split == 'val':
        dataloader = data.DataLoader(dataset,batch_size=1,
                                     shuffle=True,num_workers=config.num_workers)
    else:
        dataloader = data.DataLoader(dataset,batch_size=config.batch_size,
                                     shuffle=True,num_workers=config.num_workers)
    return dataloader

if __name__ == "__main__":
    
    train_set = LSMI(root='../data/GALAXY_synthetic',
                      split='train',image_pool=['12'],
                      image_size=256,input_type='uvl',output_type='uv')

    train_loader = data.DataLoader(train_set, batch_size=4, shuffle=False)

    for batch in train_loader:
        print("batch['fname']", batch["fname"])
        print("\nbatch['illum1']:\n", batch["illum1"])
        print("\nbatch['illum2']:\n", batch["illum2"])
        print("\nbatch['illum3']:\n", batch["illum3"])
        print("\nbatch['input'].shape:", batch["input"].shape)
        print("batch['gt'].shape:", batch["gt"].shape)
        print("batch['illum_gt'].shape:", batch["illum_gt"].shape)
        print("batch['mask'].shape:", batch["mask"].shape)
        # print("\ntorch.cat((batch['illum1'],batch['illum2']),1)")
        # print(torch.cat((batch["illum1"],batch["illum2"]),1))
        print()
        input("=== Press Enter to load next data ===")
        print()