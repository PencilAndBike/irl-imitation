import os
import cv2
import numpy as np

class ProcessCarImg(object):
  def __init__(self, f_dir):
    self._f_dir = f_dir
    self._ego_dir = f_dir+'/ego'
    self._lin_dir = f_dir+'/lin'
    self._obs_dir = f_dir+'/obs'
    self._mix_dir = f_dir+'/mix'
    
  def format_name(self, img_dir, n=3):
    for img_dir in (self._ego_dir, self._lin_dir, self._obs_dir):
      for name in os.listdir(img_dir):
        os.rename(img_dir+'/'+name, img_dir+'/'+name.zfill(7))
    
  @staticmethod
  def cmp_fname(n1, n2):
    return int(n1.split('_')[-1][:-4])-int(n2.split('_')[-1][:-4])
  
  @classmethod
  def read_imgs(cls, img_dir):
    img_names = sorted(os.listdir(img_dir), cmp=ProcessCarImg.cmp_fname)
    imgs = []
    for img_name in img_names:
      imgs.append(cv2.imread(img_dir+'/'+img_name, 0))
    return np.array(imgs)
  
  def mix(self):
    ego_imgs = ProcessCarImg.read_imgs(self._ego_dir)
    lin_imgs = ProcessCarImg.read_imgs(self._lin_dir)
    obs_imgs = ProcessCarImg.read_imgs(self._obs_dir)
    L = len(ego_imgs)
    for i, ego_img, lin_img, obs_img in zip(range(L), ego_imgs, lin_imgs, obs_imgs):
      mix_img = ego_img + lin_img + obs_img
      cv2.imwrite(self._mix_dir+'/mix_'+str(i)+'.png', mix_img)
      # np.stack((ego_img, lin_img, obj_img), axis=0)
  
  def concat(self):
    ego_imgs = ProcessCarImg.read_imgs(self._ego_dir)[:,:,:,np.newaxis]
    lin_imgs = ProcessCarImg.read_imgs(self._lin_dir)[:,:,:,np.newaxis]
    obs_imgs = ProcessCarImg.read_imgs(self._obs_dir)[:,:,:,np.newaxis]
    concated_imgs = np.concatenate((ego_imgs, lin_imgs, obs_imgs), axis=-1)
    return concated_imgs

  
  def crop_center(self, img, cent_h, cent_w):
    h, w = img.shape
    assert cent_h <= h and cent_w <= w
    return img[h/2-cent_h/2:h/2+cent_h/2, w/2-cent_w/2:w/2+cent_w/2]
  
  def crop_imgs(self, imgs, cent_h, cent_w):
    cent_imgs = []
    for img in imgs:
      cent_imgs.append(self.crop_center(img, cent_h, cent_w))
    return np.array(cent_imgs)
  
  def crop(self, cent_h, cent_w):
    ego_imgs = ProcessCarImg.read_imgs(self._ego_dir)
    lin_imgs = ProcessCarImg.read_imgs(self._lin_dir)
    obs_imgs = ProcessCarImg.read_imgs(self._obs_dir)
    return self.crop_imgs(ego_imgs, cent_h, cent_w), \
           self.crop_imgs(lin_imgs, cent_h, cent_w), \
           self.crop_imgs(obs_imgs, cent_h, cent_w)
  
  def save_imgs(self, imgs, name):
    save_path = self._f_dir+'/'+name
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    else:
      exit()
    for i, img in enumerate(imgs):
      cv2.imwrite(save_path+'/'+str(i)+'.png', img)
  
  
  def save_crop(self, cent_h=120, cent_w=120):
    crop_ego_imgs, crop_lin_imgs, crop_obs_imgs = self.crop(cent_h, cent_w)
    self.save_imgs(crop_ego_imgs, 'crop_ego')
    self.save_imgs(crop_lin_imgs, 'crop_lin')
    self.save_imgs(crop_obs_imgs, 'crop_obs')
    
  def resize(self, img_dir, size=(20,20)):
    imgs = self.read_imgs(img_dir)
    res_imgs = []
    for i, img in enumerate(imgs):
      res_imgs.append(cv2.resize(img, size))
    return res_imgs
    
  def save_resize(self, size):
    # res_ego_imgs, res_lin_imgs, res_obs_imgs, res_mix_imgs
    ego_imgs = self.resize(self._f_dir+'/crop_ego', size)
    lin_imgs = self.resize(self._f_dir+'/crop_lin', size)
    obs_imgs = self.resize(self._f_dir+'/crop_obs', size)
    mix_imgs = self.resize(self._f_dir+'/crop_mix', size)
    self.save_imgs(ego_imgs, 'resize_ego')
    self.save_imgs(lin_imgs, 'resize_lin')
    self.save_imgs(obs_imgs, 'resize_obs')
    self.save_imgs(mix_imgs, 'resize_mix')
    
    
if __name__ == '__main__':
  f_dir = '/home/pirate03/Downloads/prediction_data'
  pci = ProcessCarImg(f_dir)
  pci.save_resize((20,20))
  # mix_imgs = pci.read_imgs(f_dir+'/mix')
  # mix_imgs = pci.crop_imgs(mix_imgs, 120, 120)
  # pci.save_imgs(mix_imgs, 'crop_mix')
  # pci.save_crop()