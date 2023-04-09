#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Handlers
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Other
from pathlib import Path
from os import listdir
from time import time
from sklearn.preprocessing import OneHotEncoder
from PIL import Image, ImageOps

# Graphics
from matplotlib import pyplot as plt
import seaborn as sns

# Additional modules
from dataset_creator import generate_unshuffled_csv
from assistive_funcs import filtering_image, check_ssim, check_gmsd
from csv_dataloader import get_train_test_small_data
from image_processing import add_noise


# In[2]:


# Paths
p_main_data = Path("../data")
p_models = Path("../models")

p_scv_folder = p_main_data / "csv_files"

p_train_images = p_main_data / "train"
p_test_images = p_main_data / "test"

p_train_noised_images = p_main_data / "train_noised"
p_test_noised_images = p_main_data / "test_noised"

p_train_filtered_images = p_main_data / "train_filtered"
p_test_filtered_images = p_main_data / "test_filtered"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[3]:


# Dataset
win_size = 5
step = 3
create_dataset = 1
train_dataset_name = f"train_W{win_size}_S{step}.csv"


# In[4]:


# Train dataset
if create_dataset:
    generate_unshuffled_csv(win_size=win_size, dump_to_file=100000, step=step,
                 img_path=p_train_images,
                 datasets_path=p_scv_folder,
                 noise_imgs_path=p_train_noised_images,
                 dataset_name=train_dataset_name,
                 force_create_dataset=1)


# In[5]:


test_img_names = listdir(p_test_images)
for name in test_img_names:
    img = np.array(ImageOps.grayscale(Image.open(p_test_images / name)))
    noised_img = add_noise(img)
    Image.fromarray(noised_img.astype(np.uint8)).save(p_test_noised_images / name)


# In[14]:


class FCBlock(nn.Module):
    def __init__(self, in_len, out_len, p_dropout=False) -> None:
        super().__init__()
        if p_dropout:
            self.fc_block = nn.Sequential(
                nn.Linear(in_len, out_len),
                nn.Dropout(p_dropout),
                nn.ReLU(),
            )
        else:
            self.fc_block = nn.Sequential(
                nn.Linear(in_len, out_len),
                nn.BatchNorm1d(out_len),
                nn.ReLU(),
            )
    
    def forward(self, x):
        return self.fc_block(x)


# In[15]:


class DefaultModel(nn.Module):
    def __init__(self, in_len, out_len) -> None:
        super().__init__()
        double_in_len = in_len * 2
        triple_in_len = in_len * 3

        self.structure = nn.Sequential(
            FCBlock(in_len, double_in_len,        p_dropout=0),
            FCBlock(double_in_len, triple_in_len, p_dropout=0),
            FCBlock(triple_in_len, triple_in_len, p_dropout=0),
            FCBlock(triple_in_len, double_in_len, p_dropout=0),
            nn.Linear(double_in_len, out_len),
        )

    def forward(self, x):
        x = self.structure(x)
        return x


# In[16]:


class FitModel():
    def __init__(self, model, criterion, optimizer, scheduler,
                 p_scv_folder, train_dataset_name,
                 batch_size, device, num_epoches, normalize_data):
        # Model
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Params
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.device = device  
        
        # Folders
        self.p_scv_folder = p_scv_folder
        self.train_dataset_name = train_dataset_name
        
        # Flags
        self.normalize_data = normalize_data
        
        # Constants
        self.train_losses = []
        self.valid_losses = []
        self.images_filtered = False
        


    def _train(self, current_epoch):
        total_loss = []
        start_time = time()
        self.model.train()
        for data, targets in self.train_loader:
            
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if self.normalize_data:
                data /= 255
                targets /= 255
            
            scores = self.model(data)
            loss = self.criterion(scores, targets)
            loss.backward()
           
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss.append(loss.item())
        
        mean_total_loss = np.mean(total_loss)
        self.train_losses.append(mean_total_loss)
        print(f"Epoch: {current_epoch}/{self.num_epoches}, time: {int(time() - start_time)}s, lr = {self.scheduler.get_last_lr()}\n\tTrain loss: {mean_total_loss}")
        
    def _valid(self, current_epoch):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for data, targets in self.valid_loader:
                
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                if self.normalize_data:
                    data /= 255
                    targets /= 255
                # Forward
                scores = self.model(data)
                loss = self.criterion(scores, targets)
                total_loss.append(loss.item())
        
        mean_total_loss = np.mean(total_loss)
        self.valid_losses.append(mean_total_loss)
        print(f"\tValid loss: {mean_total_loss}")
    
    def fit(self):
        self.train_loader, self.valid_loader = get_train_test_small_data(scv_folder=self.p_scv_folder, dataset_name=self.train_dataset_name,
                                                        batch_size=self.batch_size, split=True)
        for epoch in range(self.num_epoches):
            self._train(epoch + 1)
            self._valid(epoch + 1)
            if self.scheduler is not None:
                self.scheduler.step()
                
    def plot_graph(self):
        sns.set()
        fig, (ax_train, ax_test) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        fig.suptitle('Loss')

        ax_train.set_title("Train loss")
        ax_test.set_title("Valid loss")

        ax_train.set_ylabel('Loss value')
        ax_test.set_ylabel('Loss value')

        ax_train.set_xlabel("Batch")
        ax_test.set_xlabel("Batch")

        sns.lineplot(data=self.train_losses, ax=ax_train)
        sns.lineplot(data=self.valid_losses, ax=ax_test)

        plt.show()
    
    def filtering_all_images(self):
        self.images_filtered = True
        images_names = listdir(p_test_noised_images)
        for name in images_names:
            filtering_image(self.model, p_test_filtered_images, p_test_noised_images, name, win_size, self.device, normalize_data=self.normalize_data)
        
    @staticmethod
    def _check_filtering(p_target_images, p_original_images):
        ssim_metric = []
        gmsd_metric = []
        images_names = listdir(p_target_images)
        for name in images_names:
            ssim_metric.append(round(check_ssim(p_target_images, p_original_images, name), 3))
            gmsd_metric.append(round(check_gmsd(p_target_images, p_original_images, name), 3))
        return ssim_metric, gmsd_metric
        # print(f"SSIM avg: {sum(ssim_metric) / len(ssim_metric)}")
        # print(f"GMSD avg: {sum(gmsd_metric) / len(gmsd_metric)}")
    
    def check_metrics(self, verbose=False):
        if not self.images_filtered:
            print("Warning: images weren't filtered")
        metrics_after_filtering = self._check_filtering(p_test_filtered_images, p_test_images)
        metrics_befor_filtering = self._check_filtering(p_test_noised_images, p_test_images)
        if verbose:
            print(f"After filtering\n\tmean SSIM: {np.mean(metrics_after_filtering[0]):.3f}\n\tmean GMSD: {np.mean(metrics_after_filtering[1]):.3f}")
            print(f"Before filtering\n\tmean SSIM: {np.mean(metrics_befor_filtering[0]):.3f}\n\tmean GMSD: {np.mean(metrics_befor_filtering[1]):.3f}")
            
            print(f"After filtering\n\tSSIM: {metrics_after_filtering[0]}\n\tGMSD: {metrics_after_filtering[1]}")
            print(f"Before filtering\n\tSSIM: {metrics_befor_filtering[0]}\n\tGMSD: {metrics_befor_filtering[1]}")
        else:
            print(f"After filtering\n\tSSIM: {np.mean(metrics_after_filtering[0]):.3f}\n\tGMSD: {np.mean(metrics_after_filtering[1]):.3f}")
            print(f"Before filtering\n\tSSIM: {np.mean(metrics_befor_filtering[0]):.3f}\n\tGMSD: {np.mean(metrics_befor_filtering[1]):.3f}")
        


# In[19]:


# Hyperparameters 
learning_rate = 0.1
num_epoches = 20
batch_size = 1024
normalize_data = False


model = DefaultModel(in_len=(win_size ** 2), out_len=1).to(device=device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1) # StepLR
# scheduler = None

fit_model = FitModel(model, criterion, optimizer, scheduler,
                     p_scv_folder, train_dataset_name,
                     batch_size, device, num_epoches,
                     normalize_data)


# In[20]:


fit_model.fit()


# In[21]:


fit_model.plot_graph()


# In[22]:


fit_model.filtering_all_images()


# In[23]:


fit_model.check_metrics(verbose=True)

