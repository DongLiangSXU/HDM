import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.utils.data as data
from torch.utils.data import DataLoader
import torch,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
import torch,warnings
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from PIL import Image
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
    'hdm':DR_Net_phy(3,3)
}

start_time=time.time()
T=opt.steps
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr


import dehaze
def tran_mask(img):
    width,height = img.size

    img = img.resize((width//4, height//4),Image.ANTIALIAS)

    # img.save('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/smallhaze.png')


    haze = np.asarray(img)
    I = np.asarray(img) / 255
    dark = dehaze.DarkChannel(I, 15);
    A = dehaze.AtmLight(I, dark);

    dc, a = dehaze.get_dc_A(I, 111, 0.001, 0.95, 0.80)

    A = A - A + a
    te = dehaze.TransmissionEstimate(I, A, 15);
    t = dehaze.TransmissionRefine(haze, te);

    tfile = t
    # img.save('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/haze.png')
    # print(tfile.shape)
    ht,wt = tfile.shape[0],tfile.shape[1]
    maxv = np.max(tfile)
    minv = np.min(tfile)
    tfile_norm = (tfile-minv)/(maxv-minv)
    mask1 = np.where(tfile_norm>= 0.8, np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask2 = np.where((tfile_norm>= 0.6) & (tfile_norm<0.8), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask3 = np.where((tfile_norm>= 0.4) & (tfile_norm<0.6), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask4 = np.where((tfile_norm>= 0.2) & (tfile_norm<0.4), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask5 = np.where((tfile_norm>= 0.0) & (tfile_norm<0.2), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    # B x 3 x 1 x 25 x 25
    mask1 = np.resize(mask1,[1,1,ht,wt])
    mask2 = np.resize(mask2, [1,1, ht, wt])
    mask3 = np.resize(mask3, [1,1, ht, wt])
    mask4 = np.resize(mask4, [1,1, ht, wt])
    mask5 = np.resize(mask5, [1,1, ht, wt])
    mask1 = torch.from_numpy(np.ascontiguousarray(mask1)).to(torch.float)
    mask2 = torch.from_numpy(np.ascontiguousarray(mask2)).to(torch.float)
    mask3 = torch.from_numpy(np.ascontiguousarray(mask3)).to(torch.float)
    mask4 = torch.from_numpy(np.ascontiguousarray(mask4)).to(torch.float)
    mask5 = torch.from_numpy(np.ascontiguousarray(mask5)).to(torch.float)
    allmask = torch.cat([mask1,mask2,mask3,mask4,mask5],dim=0)
    # vutils.save_image(allmask.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/mask.png')
    # print(allmask.shape)
    # exit(-1)
    return allmask


class SOTS_Dataset(data.Dataset):
    def __init__(self,path,train,format='.png'):
        super(SOTS_Dataset,self).__init__()
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')

    def __getitem__(self, index):

        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]

        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))

        haze,clear,pldata=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        allmask = tran_mask(pldata.convert("RGB"))

        return haze,clear,allmask,clear_name

    def augData(self,data,target):
        pldata = data
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return data ,target,pldata

    def __len__(self):
        return len(self.haze_imgs)



def test(net,loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims=[]
    psnrs=[]
    for i ,(inputs,targets,_,mask) in enumerate(loader_test):
        inputs=inputs.cuda();targets=targets.cuda();mask = mask.cuda();_ = _.cuda()
        pred=net(inputs,mask)

        ssim1=ssim(pred,targets).item()
        psnr1=psnr(pred,targets)

        ssims.append(ssim1)
        psnrs.append(psnr1)


    print('SOT Test result: SSIM: {:.4f}, PSNR: {:.4f}'.format(
        np.mean(ssims),
        np.mean(psnrs)
    ))


# 35.8

if __name__ == "__main__":

    net=models_['hdm']
    net=net.to(opt.device)

    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        cudnn.benchmark=True

    net.load_state_dict(torch.load('./trained_models/hdm_haze_sots.pk')['model'])

    loader_test = DataLoader(dataset=SOTS_Dataset('./SOT', train=False), batch_size=1,shuffle=False)

    with torch.no_grad():
        test(net, loader_test)



