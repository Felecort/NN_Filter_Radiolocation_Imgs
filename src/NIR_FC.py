#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Handlers
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps

# Pytorch
import torch
import torch.nn as nn  # NN; networks (CNN, RNN, losses)
import torch.optim as optim  # Optimizers (Adam, Adadelta, Adagrad)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset  # Dataset manager
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

# Other
import datetime
import os
from tqdm import tqdm
from pathlib import Path
from os import listdir
from fastaniso import anisodiff
from time import time


# Graphics
from matplotlib import pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim


# Additional modules
from dataset_creator import generate_csv
from assistive_funcs import filtering_image, check_ssim, check_gmsd, convert_to_grayscale, get_dataset_name
from csv_dataloader import get_train_test_data
from math import floor


# In[2]:


# Paths
p_main_data = Path("../data")
p_models = Path("../models")

p_scv_folder = p_main_data / "csv_files" # datasets_path
p_img = p_main_data / "images"

p_noised_imgs = p_main_data / "FC_imgs_with_noise"

p_filtered_images = p_main_data / "FC_filtered_images"

p_gray_images = p_main_data / "gray_images"


# In[3]:


# Hyperparameters 
learning_rate = 0.01
num_epoches = 9
batch_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"{dataset_name = }\n{device = }")
device


# In[4]:


# Dataset
win_size = 7
step = 5
# create_dataset = 1
# if create_dataset:
#     generate_csv(win_size=win_size, dump_to_file=5000, step=step, force_create_dataset=1)


# In[5]:


win_size = 7
step = 5

dataset_name = get_dataset_name(win_size, step, p_scv_folder) #r"W5_S1_L3696640.csv"
dataset_name


# In[6]:


# from google.colab import drive
# drive.mount('/content/gdrive/')
# !unzip -q /content/gdrive/MyDrive/NIR/data/FC_data/W5_S5_L146410.zip


# In[7]:


class DefaultModel(nn.Module):
    def __init__(self, in_len, out_len) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.hid_n = 20
        
        self.fcs = nn.Sequential(
            nn.Linear(self.in_len, self.in_len),
            nn.BatchNorm1d(self.in_len),
            nn.ReLU(),
            nn.Linear(self.in_len, self.hid_n * 2),
            nn.BatchNorm1d(self.hid_n * 2),
            nn.ReLU(),
            nn.Linear(self.hid_n * 2, self.hid_n * 2),
            nn.BatchNorm1d(self.hid_n * 2),
            nn.ReLU(),
            nn.Linear(self.hid_n * 2, self.hid_n),
            nn.BatchNorm1d(self.hid_n),
            nn.ReLU(),
            nn.Linear(self.hid_n, self.out_len),
        )

    def forward(self, x):
        x = self.fcs(x)
        return x


# In[8]:


model = DefaultModel(in_len=(win_size ** 2), out_len=1).to(device=device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[9]:


losses = []
losses_append = losses.append

valid_losses = []
valid_losses_append = valid_losses.append

for epoch in range(num_epoches):
    model.train()
    train_loader, test_loader = get_train_test_data(scv_folder=p_scv_folder, dataset_name=dataset_name, batch_size=batch_size, train_size=0.8)
    for batch_ind, (data, targets) in tqdm(enumerate(train_loader)):
        # Data on cuda
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Forward
        scores = model(data) # Equal to model.forward(data)
        loss = criterion(scores, targets)
        if batch_ind % 3 == 0:
            losses_append(loss.item())
        # Backprop
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for batch_ind, (data, targets) in tqdm(enumerate(test_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # Forward
            scores = model(data) # Equal to model.forward(data)
            loss = criterion(scores, targets)
            valid_losses_append(loss.item())


# In[10]:


sns.set()
fig, (ax_train, ax_test) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle('Loss')

ax_train.set_title("Train loss")
ax_test.set_title("Test loss")

ax_train.set_ylabel('Loss value')
ax_test.set_ylabel('Loss value')

ax_train.set_xlabel("k * Batch")
ax_test.set_xlabel("k * Batch")

sns.lineplot(data=losses, ax=ax_train)
sns.lineplot(data=valid_losses, ax=ax_test)

plt.show()


# In[11]:


list_images = listdir(p_noised_imgs)
for img_name in list_images:
    filtering_image(model, p_filtered_images, p_noised_imgs, img_name, win_size, device)


# In[12]:


images_names = listdir(p_noised_imgs)
for name in images_names:
    check_ssim(p_filtered_images, p_gray_images, name)


# In[13]:


images_names = listdir(p_noised_imgs)
for name in images_names:
    check_gmsd(p_filtered_images, p_gray_images, name)


# In[14]:


# p_img = p_raw_image / list_images[2]
# img_arr = np.array(ImageOps.grayscale(Image.open(p_img)))
# niters = [1, 2, 5, 10, 20]
# kappa = [1, 5, 10, 50, 100]
# for i in niters:
#     for k in kappa:
#         img_filtered = anisodiff(img_arr, niter=i, kappa=k)
#         img_filtered = img_filtered.astype(np.uint8)
#         Image.fromarray(img_filtered).save(f"{p_img}_I{i}_K{k}.jpg")

