# terrain_traversability/utils/simclr.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

def get_simclr_augmentations(size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128):
        super().__init__()
        self.encoder = getattr(models, base_model)(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create labels on the same device as the input tensors
        device = z1.device
        labels = torch.cat([torch.arange(z1.size(0), device=device), torch.arange(z2.size(0), device=device)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        return nn.CrossEntropyLoss()(logits, labels)