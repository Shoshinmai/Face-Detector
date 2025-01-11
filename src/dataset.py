import torch
import numpy as np
import pandas as pd
import config
import util
import cv2

from torch.utils.data import Dataset, DataLoader

def train_test_split(csv_path , split):
    df_data = pd.read_csv(csv_path)
    len_data = len(df_data)
    valid_split = int(len_data * split)
    train_split = int(len_data - valid_split)
    training_sample = df_data.iloc[:train_split][:]
    valid_sample = df_data.iloc[-valid_split:][:]
    return training_sample, valid_sample

class FaceKeypointDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 224
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        image = cv2.imread(f"{self.path}/{self.data.iloc[index][0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        image = cv2.resize(image, (self.resize, self.resize))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype='float32')
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints * [self.resize/orig_w, self.resize/orig_h]

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float)
        }
    
training_samples, valid_samples = train_test_split(r"D:\AI dev\ML\project\facial detection RGB\input\training.csv",
                                                   config.TEST_SPLIT)

train_data = FaceKeypointDataset(training_samples, 
                                 r"D:\AI dev\ML\project\facial detection RGB\input\training")
valid_data = FaceKeypointDataset(valid_samples, 
                                 r"D:\AI dev\ML\project\facial detection RGB\input\training")

train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)
valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)
print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

if config.SHOW_DATASET_PLOT:
    util.dataset_keypoints_plot(valid_data)