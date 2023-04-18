# Imports
from model_arch_dropout import *
from augment import *
import os
import cv2
import tqdm
import hashlib
import requests

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#from numba import jit

from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import find_objects, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from scipy.optimize import linear_sum_assignment

from seg_net import SegNet

# @title Download and normalize data
filenames = ["cells_train.npz",
             "cells_test.npz"]
urls = ["https://osf.io/z3h78/download",
        "https://osf.io/ft5p3/download"]
expected_md5s = ["85e1fe2ee8d936c1083d62563d79d958",
                 "e8f789abe20a7efde806d9ba03d20fd7"]

for fname, url, expected_md5 in zip(filenames, urls, expected_md5s):
  if not os.path.isfile(fname):
    try:
      r = requests.get(url)
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      elif hashlib.md5(r.content).hexdigest() != expected_md5:
        print("!!! Data download appears corrupted !!!")
      else:
        with open(fname, "wb") as fid:
          fid.write(r.content)

cells_train = np.load('cells_train.npz', allow_pickle=True)['arr_0'].item()
cells_test = np.load('cells_test.npz', allow_pickle=True)['arr_0'].item()
imgs_train = np.array(cells_train['imgs']).transpose(0, 3, 1, 2)
masks_train = np.array(cells_train['masks'])
imgs_test = np.array(cells_test['imgs']).transpose(0, 3, 1, 2)
masks_test = np.array(cells_test['masks'])

num_images = 254
res_x = 608
res_y = 608

def dataloader_airsim():#277
    imgs = np.zeros((num_images,res_x,res_y,3)).astype(np.uint8)
    masks = np.zeros((num_images,res_x,res_y,3)).astype(np.uint8)
    j = 1
    for i in range(277):
      if cv2.imread("E:\\AirSim\\PythonClient\\multirotor\\custom_seg\\input\\airsim_drone_608_200\\" + str(i) + "_surfaceNormal.png") is None:
        continue
      imgs[j-1] = cv2.imread("E:\\AirSim\\PythonClient\\multirotor\\custom_seg\\input\\airsim_drone_608_200\\" + str(i) + "_surfaceNormal.png")
      masks[j-1] = cv2.imread("E:\\AirSim\\PythonClient\\multirotor\\custom_seg\\input\\airsim_drone_608_200\\" + str(i) + "_combinedMask.png")
      j += 1
    return imgs, masks

def binarize(masks):
  index = masks.shape
  binary_masks = np.zeros((num_images,res_x,res_y)).astype(np.uint8)
  for i in range(index[0]):
      gray_mask = cv2.cvtColor(masks[i], cv2.COLOR_BGR2GRAY)
      _, binary_masks[i] = cv2.threshold(gray_mask, 0, 1, cv2.THRESH_OTSU)
  return binary_masks

imgs_train_temp, masks_train_temp = dataloader_airsim()
imgs_train = imgs_train_temp.transpose(0,3,1,2)
masks_train = binarize(masks_train_temp)


# we are going to normalize the images so their pixel values mostly fall between 0 and 1
# this is helpful if you have images on a variety of scales
# we will also return the images as float32 <- the data type that is fast for GPU computation
def normalize99(img):
  """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
  X = img.copy()
  xmin = np.amin(X)
  xmax = np.amax(X)
  X = (X - xmin) / (xmax - xmin)
  return X.astype(np.float32)


imgs_train = np.array([normalize99(img) for img in imgs_train])
imgs_test = np.array([normalize99(img) for img in imgs_test])

img_batch, lbl_batch, scale = random_rotate_and_resize(imgs_train[:8], masks_train[:8])

'''
plt.figure(figsize=(16, 12))
for j in range(8):
  plt.subplot(8, 3, 3*j + 1)
  plt.imshow(img_batch[j, 0])
  plt.title('channel 1 - cytoplasm')
  plt.axis('off')

  plt.subplot(8, 3, 3*j + 2)
  plt.imshow(img_batch[j, 1])
  plt.title('channel 2 - nuclei')
  plt.axis('off')

  plt.subplot(8, 3, 3*j + 3)
  plt.imshow(lbl_batch[j, 0])
  plt.title('cell masks')
  plt.axis('off')
plt.tight_layout()
plt.show()
'''
labels_train = np.zeros((len(masks_train), 2,
                         masks_train.shape[-2],
                         masks_train.shape[-1]),
                        np.int64)
labels_train[:, 0] = masks_train == 0
labels_train[:, 1] = masks_train > 0

labels_test = np.zeros((len(masks_test), 2,
                        masks_test.shape[-2],
                        masks_test.shape[-1]),
                       np.int64)
labels_test[:, 0] = masks_test == 0
labels_test[:, 1] = masks_test > 0

kernel_size = 3
nbase = [3, 32, 64, 128, 256]  # number of channels per layer
#nbase = [3, 128, 256, 512, 1024]  # number of channels per layer
nout = 1  # number of outputs

net = Unet(nbase, nout, kernel_size)
#net = SegNet(3, nout)
#print(net)
# put on GPU here if you have it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(torch.cuda.get_device_name(0))
net.to(device);  # remove semi-colon to see net structure

# train the network
# parameters related to training the network
batch_size = 1 #48 # number of images per batch -- amount of required memory
              # for training will increase linearly in batchsize
### you will want to increase n_epochs!
n_epochs = 1000  # number of times to cycle through all the data during training
learning_rate = 0.1 # initial learning rate
weight_decay = 1e-5 # L2 regularization of weights
momentum = 0.9 # how much to use previous gradient direction
n_epochs_per_save = 25 # how often to save the network
val_frac = 0.0005 # what fraction of data to use for validation

# where to save the network
# make sure to clean these out every now and then, as you will run out of space
now = datetime.now()
timestamp = now.strftime('%Y%m%dT%H%M%S')

# split into train and validation datasets
n_val = int(len(imgs_train) * val_frac)
n_train = len(imgs_train) - n_val
np.random.seed(10)
iperm = np.random.permutation(len(imgs_train))
train_data, val_data = imgs_train[iperm[:n_train]], imgs_train[iperm[n_train:]]
train_labels, val_labels = labels_train[iperm[:n_train]], labels_train[iperm[n_train:]]
train_masks, val_masks = masks_train[iperm[:n_train]], masks_train[iperm[n_train:]]


# gradient descent flavor
optimizer = torch.optim.SGD(net.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=0.9)
# set learning rate schedule
LR = np.linspace(0, learning_rate, 10)
if n_epochs > 250:
    LR = np.append(LR, learning_rate*np.ones(n_epochs-100))
    for i in range(10):
        LR = np.append(LR, LR[-1]/2 * np.ones(10))
else:
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

# store loss per epoch
epoch_losses = np.zeros(n_epochs)
epoch_losses[:] = np.nan

# when we last saved the network
saveepoch = None

#imgs_vis = imgs_train.cpu()
#imgs_vis = imgs_vis.detach().numpy()
imgs_vis = 255*imgs_train
imgs_vis = imgs_vis.astype(np.uint8).transpose(0,2,3,1)
cv2.imshow('Network Input', imgs_vis[0])

# loop through entire training data set nepochs times
for epoch in range(n_epochs):
  net.train() # put in train mode (affects batchnorm)
  epoch_loss = 0
  iters = 0
  for param_group in optimizer.param_groups:
    param_group['lr'] = LR[epoch]
  with tqdm.tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{n_epochs}", unit='img') as pbar:
    # loop through each batch in the training data
    for ibatch in np.arange(0, n_train, batch_size):
      # augment the data
      inds = np.arange(ibatch, min(n_train, ibatch+batch_size))
      imgs, lbls, _ = random_rotate_and_resize(train_data[inds],train_labels[inds])
      #imgs = train_data[inds]
      #lbls = train_labels[inds]

      #index = 0

      imgs_vis = 255*imgs
      #imgs_vis = imgs_vis.astype(np.uint8).reshape(65,144,256,3)
      imgs_vis_ind = imgs_vis.shape
      index = np.random.randint(0, imgs_vis_ind[0])
      imgs_vis = imgs_vis.astype(np.uint8).transpose(0,2,3,1)
      cv2.imshow('Network Input', imgs_vis[index])

      # transfer to torch + GPU
      imgs = torch.from_numpy(imgs).to(device=device)
      lbls = torch.from_numpy(lbls).to(device=device)

      # compute the loss
      #imgs_vis = imgs.cpu()
      #imgs_vis = imgs_vis.detach().numpy()
      #imgs_vis = 255*imgs_vis
      #imgs_vis = imgs_vis.astype(np.uint8).reshape(65,224,224,3)
      #cv2.imshow('Network Input', imgs_vis[0])


  
      y = net(imgs)

      y_vis = y.cpu()
      y_vis = y_vis.detach().numpy()
      y_vis = 255*y_vis
      y_vis = y_vis.astype(np.uint8)
      cv2.imshow('Network Prediction 1', y_vis[index,0])
      #cv2.imshow('Network Prediction 2', y_vis[0,1])

      lbls_vis = lbls.cpu()
      lbls_vis = lbls_vis.detach().numpy()
      lbls_vis = 255*lbls_vis
      lbls_vis = lbls_vis.astype(np.uint8)
      cv2.imshow('Ground Truth 1', lbls_vis[index,1])
      #cv2.imshow('Ground Truth 2', lbls_vis[0,0])
      
      cv2.waitKey(1)

      loss = criterion(y[:,0], lbls[:, 1].float())
      epoch_loss += loss.item()
      pbar.set_postfix(**{'loss (batch)': loss.item()})
      # gradient descent
      optimizer.zero_grad()
      loss.backward()
      #nn.utils.clip_grad_value_(net.parameters(), 0.1)
      optimizer.step()
      iters+=1
      pbar.update(imgs.shape[0])

    epoch_losses[epoch] = epoch_loss
    pbar.set_postfix(**{'loss (epoch)': epoch_loss})  #.update('loss (epoch) = %f'%epoch_loss)

  # save checkpoint networks every now and then
  if epoch % n_epochs_per_save == 0:
    print(f"\nSaving network state at epoch {epoch+1}")
    saveepoch = epoch
    savefile = f"unet_epoch{saveepoch+1}.pth"
    net.save_model(savefile)
print(f"\nSaving network state at epoch {epoch+1}")
net.save_model(f"unet_epoch{epoch+1}.pth")