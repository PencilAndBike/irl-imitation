import os
import cv2
import numpy as np
import shutil

class ProcessCarImg(object):
  def __init__(self, f_dir=None):
    # pass
    self._f_dir = f_dir
    # # self._ego_dir = f_dir+'/ego'
    self._ego_dir = None
    self._lin_dir = f_dir+'/lin'
    self._obs_dir = f_dir+'/obs'
    # self._mix_dir = f_dir+'/mix'
    
  def format_name(self, img_dir, n=3):
    for img_dir in (self._ego_dir, self._lin_dir, self._obs_dir):
      for name in os.listdir(img_dir):
        os.rename(img_dir+'/'+name, img_dir+'/'+name.zfill(7))
    
  @staticmethod
  def cmp_fname(n1, n2):
    return int(n1.split('_')[-1][:-4])-int(n2.split('_')[-1][:-4])
  
  def read_imgs(self, img_dir):
    img_names = sorted(os.listdir(img_dir), cmp=ProcessCarImg.cmp_fname)
    imgs = []
    for img_name in img_names:
      imgs.append(cv2.imread(img_dir+'/'+img_name, 0))
    return np.array(imgs)
  
  def mix(self, lin_dir, obs_dir, mix_dir):
    # ego_imgs = ProcessCarImg.read_imgs(self._ego_dir)
    lin_imgs = self.read_imgs(lin_dir)
    obs_imgs = self.read_imgs(obs_dir)
    L = len(lin_imgs)
    for i, lin_img, obs_img in zip(range(L), lin_imgs, obs_imgs):
      mix_img = np.stack((lin_img, obs_img, np.zeros(lin_img.shape)), axis=-1)
      cv2.imwrite(mix_dir+'/'+str(i).zfill(4)+'.png', mix_img)
  
  def mix_set(self, fdir):
    for eps_name in sorted(os.listdir(fdir)):
      lin_dir = fdir+'/'+eps_name+'/lin'
      obs_dir = fdir+'/'+eps_name+'/obs'
      mix_dir = fdir+'/'+eps_name+'/mix'
      os.mkdir(mix_dir)
      self.mix(lin_dir, obs_dir, mix_dir)
      
  def concat(self):
    # ego_imgs = ProcessCarImg.read_imgs(self._ego_dir)[:,:,:,np.newaxis]
    lin_imgs = self.crop_imgs(self.read_imgs(self._lin_dir), cent_h=256, cent_w=256)[:,:,:,np.newaxis]
    obs_imgs = self.crop_imgs(self.read_imgs(self._obs_dir), cent_h=256, cent_w=256)[:,:,:,np.newaxis]
    concated_imgs = np.concatenate((lin_imgs, obs_imgs), axis=-1)
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
    mix_imgs = ProcessCarImg.read_imgs(self._mix_dir)
    return self.crop_imgs(ego_imgs, cent_h, cent_w), \
           self.crop_imgs(lin_imgs, cent_h, cent_w), \
           self.crop_imgs(obs_imgs, cent_h, cent_w), \
           self.crop_imgs(mix_imgs, cent_h, cent_w)
  
  def save_imgs(self, imgs, name):
    save_path = self._f_dir+'/'+name
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    else:
      exit()
    for i, img in enumerate(imgs):
      cv2.imwrite(save_path+'/'+str(i)+'.png', img)
  
  def save_crop(self, cent_h=200, cent_w=200, save_dir=None):
    crop_ego_imgs, crop_lin_imgs, crop_obs_imgs, crop_mix_imgs = self.crop(cent_h, cent_w)
    self.save_imgs(crop_ego_imgs, 'crop_ego')
    self.save_imgs(crop_lin_imgs, 'crop_lin')
    self.save_imgs(crop_obs_imgs, 'crop_obs')
    self.save_imgs(crop_mix_imgs, 'crop_mix')
    
    
  def resize_save_imgs(self, input_dir, output_dir, kernel=(7,7), shape=(128,128)):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    img_names = sorted(os.listdir(input_dir))
    res_imgs = []
    for img_name in img_names:
      img = cv2.imread(input_dir+"/"+img_name, 0)/100.0*255
      res_imgs.append(cv2.resize(cv2.dilate(img, kernel, 1), shape, interpolation=cv2.INTER_AREA))
    for res_img, img_name in zip(res_imgs, img_names):
      cv2.imwrite(output_dir+'/'+img_name, res_img)

  def resize_save_imgs_set(self, input_fdir, output_fdir):
    for eps_name in sorted(os.listdir(input_fdir)):
      obs_dir = input_fdir+'/'+eps_name+'/obs'
      lin_dir = input_fdir+'/'+eps_name+'/lin'
      obs_out_dir = output_fdir+'/'+eps_name+'/obs'
      lin_out_dir = output_fdir+'/'+eps_name+'/lin'
      self.resize_save_imgs(obs_dir, obs_out_dir)
      self.resize_save_imgs(lin_dir, lin_out_dir)
      
      
  def cp_status_set(self, input_fdir, output_fdir):
    if not os.path.exists(output_fdir):
      os.mkdir(output_fdir)
    eps_names = sorted(os.listdir(input_fdir))
    for eps_name in eps_names:
      shutil.copy(input_fdir+'/'+eps_name+'/status.txt', output_fdir+'/'+eps_name+'/status.txt')
  
  
  def make_data(self, input_fdir, output_fdir):
    if not os.path.exists(output_fdir):
      os.mkdir(output_fdir)
    self.resize_save_imgs_set(input_fdir, output_fdir)
    self.cp_status_set(input_fdir, output_fdir)

  
  def zfill_eps_names(self, f_dir):
    eps_names = os.listdir(f_dir)
    for eps_name in eps_names:
      shutil.move(f_dir+'/'+eps_name, f_dir+'/'+eps_name.zfill(4))

  
  def find_wrong_imgs(self, f_dir):
    eps_names = os.listdir(f_dir)
    for eps_name in eps_names:
      obs_dir = f_dir+'/'+eps_name+'/obs'
      obs_imgs = self.read_imgs(obs_dir)
      for i, img in enumerate(obs_imgs):
        if img is None:
          print 'eps {}: {}'.format(eps_name, i)
          break
      
      
if __name__ == '__main__':
  input_fdir = '/home/pirate03/Downloads/carsim/train'
  output_fdir = '/home/pirate03/Downloads/carsim/test'
  pci = ProcessCarImg('ddd')
  # pci.find_wrong_imgs(input_fdir)
  # pci.make_data(input_fdir, output_fdir)
  
  # fdir = '/home/pirate03/Downloads/carsim/resized_train'
  pci.mix_set(output_fdir)
  
  
  
  
  # eps_names = sorted(os.listdir(fdir))
  # for eps_name in eps_names:
  #   rec_dir = f_dir+'/'+eps_name
  #   pci = ProcessCarImg(rec_dir)
  #   pci.save_crop(cent_h=256, cent_w=256)
  
  
  # pci.save_resize((20,20))
  # mix_imgs = pci.read_imgs(f_dir+'/mix')
  # mix_imgs = pci.crop_imgs(mix_imgs, 120, 120)
  # pci.save_imgs(mix_imgs, 'crop_mix')
  # pci.save_crop()