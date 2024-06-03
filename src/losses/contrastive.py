import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm
import os
import pandas as pd
import einops
from scipy.stats import mode
import numpy as np


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(exp_similarities, offset=-1)

        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities[:,:-1], dim=0) - positives
        
        # Calculate NT-Xent loss
        loss = -torch.log(positives / denominator).mean()

        return loss


class LS_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(LS_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        
        # Normalize the feature vectors
        features_normalized = torch.nn.functional.normalize(features, p=2, dim=-1)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        lower_diag = torch.diagonal(exp_similarities, offset=-1)
        
        # The numerator is the sum of shifted left and right of the positive pairs
        numerator = lower_diag[1:] + lower_diag[:-1]
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities[:,:-2], dim=0) - lower_diag[:-1]\
                + (torch.sum(exp_similarities[:,1:-1], dim=0)  - (lower_diag[1:] + lower_diag[:-1]))
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(numerator / denominator).mean()
        
#         print("Similarities: ", similarities)
#         print("Exp Similarities: ", exp_similarities)
#         print("Numerator: ", numerator)
#         print("Denominator: ", denominator)
        
        return loss
    
class NN_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NN_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(exp_similarities, offset=-1)
        
        
#         # Normalize the feature vectors
#         features_normalized2 = F.normalize(features2, dim=-1, p=2)

#         # Calculate the cosine similarity matrix
#         similarities2 = torch.matmul(features_normalized2, features_normalized2.T)
        
#         exp_similarities2 = torch.exp(similarities2 / self.temperature)
        
#         # Removing the similarity of a window with itself i.e main diagonal
#         exp_similarities2 = exp_similarities2 - torch.diag(exp_similarities2.diag())        

#         # Lower diagonal elements represent positive pairs
#         positives2 = torch.diagonal(exp_similarities2, offset=-1)
        
#         # The denominator is the sum of the column vectors minus the positives
#         denominator = torch.sum(exp_similarities[:,:-1], dim=0) - positives2
        
        
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(positives).mean()

        return loss

class RAN_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(RAN_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features, features2):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(exp_similarities, offset=-1)
        
        
        # Normalize the feature vectors
        features_normalized2 = F.normalize(features2, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities2 = torch.matmul(features_normalized2, features_normalized2.T)
        
        exp_similarities2 = torch.exp(similarities2 / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities2 = exp_similarities2 - torch.diag(exp_similarities2.diag())        

        # Lower diagonal elements represent positive pairs
        positives2 = torch.diagonal(exp_similarities2, offset=-1)
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities2[:,:-1], dim=0) - positives2
        
        
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(positives/denominator).mean()

        return loss


# Define custom dataset class
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get original image and label from the original dataset
        original_image, label = self.original_dataset[idx]

        # Apply transformations to the original image to get augmented image
        augmented_image = self.transform(original_image)

        return original_image, augmented_image, label

# Define transformations for augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(size=64),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])


class SPAT_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(SPAT_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features, features2):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)
        features_normalized2 = F.normalize(features2, dim=-1, p=2)
        

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized2.T)
        
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        
        # main diagonal elements represent positive pairs
        numerator = torch.diag(exp_similarities.diag())
        
        
        # Removing the similarity of a window with its augmentation i.e main diagonal
        exp_similarities = exp_similarities - numerator
        

        # The denominator is the sum of the column vectors minus the positives
        numerator = torch.sum(numerator, dim=0)
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities, dim=0)
        
        # Calculate NT-Xent loss
        loss = -torch.log(numerator/denominator).mean()

        return loss
    
class MARGIN_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5, margin=5):
        super(MARGIN_LOSS, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, sim_vector):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        
        


        # Removing the similarity of a window with itself i.e main diagonal
        similarities = similarities - torch.diag(similarities.diag())  
    
        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(similarities, offset=-1)
        

        # The denominator is the sum of the column vectors minus the positives
        denominator = similarities[:,:-1]
        

        # The denominator is the sum of the column vectors minus the positives
        new_denominator = (1-sim_vector)*0.5*( denominator )**2 + \
                      sim_vector*0.5*torch.max(torch.tensor(0), (self.margin - (denominator)))**2
        
        exp_numerator = torch.exp(positives / self.temperature)
        

        exp_denominator = torch.exp(new_denominator)


        exp_denominator = torch.sum(exp_denominator, dim=0) - exp_numerator
        
        # Calculate NT-Xent loss
        loss = -torch.log(exp_numerator / exp_denominator).mean()
        
        return loss
    
class LS_MARGIN_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5, margin=5):
        super(LS_MARGIN_LOSS, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, sim_vector):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        # Lower diagonal elements represent positive pairs
        lower_diag = torch.diagonal(similarities, offset=-1)
        exp_numerator = torch.exp(lower_diag[1:] / self.temperature) + torch.exp(lower_diag[:-1] / self.temperature)
    
    
        # The denominator is the sum of the column vectors minus the positives
        new_similarities = -(1-sim_vector)*0.5*( similarities )**2 + \
                    sim_vector*0.5*torch.max(torch.tensor(0), (self.margin - (similarities)))**2    

        # Remove negative and introduced gamma for double margin to avoid NaN values
        exp_sim = torch.exp(0.05*new_similarities  / self.temperature)
    
        exp_similarities = exp_sim - torch.diag(exp_sim.diag())
    
        # The denominator is the sum of the column vectors minus the positives
        exp_denominator = torch.sum(exp_similarities[:,:-2], dim=0) - torch.exp(lower_diag[:-1] / self.temperature)\
                + (torch.sum(exp_similarities[:,1:-1], dim=0)  - (exp_numerator))
    
        # Calculate NT-Xent loss
        loss = -torch.log(exp_numerator / (exp_denominator + exp_numerator)).mean()
    
        return loss


