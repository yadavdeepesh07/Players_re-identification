from torchvision import models, transforms
import torch.nn as nn
import torch

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

def extract_features(cropped_image_tensor):
    with torch.no_grad():
        return resnet(cropped_image_tensor.unsqueeze(0)).squeeze(0)
