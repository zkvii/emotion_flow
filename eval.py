import torch
import torch.nn as nn
from torchmetrics.text import cer, wer
from torch.utils.data import DataLoader

if __name__ == '__main__':
    print('eval')