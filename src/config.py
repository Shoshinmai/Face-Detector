import torch

ROOT_PATH = r'..\input'
OUTPUT_PATH = r'..\outputs'

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
TEST_SPLIT = 0.1
SHOW_DATASET_PLOT = True