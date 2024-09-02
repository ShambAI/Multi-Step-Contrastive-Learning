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
import numpy


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
    
class TripletLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, compared_length, nb_random_samples, negative_penalty, output_size):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.output_size = output_size

    def forward(self, batch, encoder, train, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))
        
#         print("length: ", length)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)
        
#         print("samples:", samples)

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)
#         print("length_pos_neg: ", length_pos_neg)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
#         print("random_length: ", random_length)
        
        length_pos_neg = random_length
        
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors
#         print("beginning_batches: ", beginning_batches)
        

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  
#         print("beginning_samples_pos: ", beginning_samples_pos)
        
        # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
#         print("beginning_positive: ", beginning_positive)
        
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg
#         print("end_positive: ", end_positive)
        
        

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )
        
        default_rep = torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        )
        
        default_rep_transposed = default_rep.transpose(1, 2)

        representation = encoder(default_rep_transposed)  # Anchors representations
        
        positive_rep = torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        )
        positive_rep_transposed = positive_rep.transpose(1, 2)
        positive_representation = encoder(positive_rep_transposed)  # Positive samples representations

        size_representation = representation.size(1)
        size_posrepresentation = positive_representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        
        
#         print(representation.shape)
#         print(positive_representation.shape)
        
        
        
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.reshape(batch_size, self.output_size, size_representation),
            positive_representation.reshape(batch_size, size_posrepresentation, self.output_size)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            
            negative_rep = torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            
            negative_rep_transposed = negative_rep.transpose(1, 2)
            negative_representation = encoder(negative_rep_transposed)
            
#             print(negative_representation.shape)
            
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.reshape(batch_size, self.output_size, size_representation),
                    negative_representation.reshape(
                        batch_size, size_representation, self.output_size
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


