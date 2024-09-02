import torch
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *
import torch
import src.config, src.utils, src.models, src.hunt_data
from src.losses.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from src.loader.dataloader import SequentialRandomSampler, FlattenedDataset, STFTDataset, SLEEPDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import wandb
import argparse
import random
import math
import pickle


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import src.config, src.utils, src.models, src.data
import torchvision
from torch.utils.data import random_split

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model(seed):
    set_seed(seed)
    model = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features'])
    return model

def load_balanced_dataset(dataset, class_counts):
    # Initialize dictionary to store indices of each class
    class_indices = {label: [] for label in class_counts.keys()}
    
    # Populate class_indices with indices of each class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item()  # Ensure the label is a scalar
        if label in class_indices:
            class_indices[label].append(idx)

    # Ensure each class has the required number of instances
    balanced_indices = []
    for label, count in class_counts.items():
        if len(class_indices[label]) >= count:
            balanced_indices.extend(random.sample(class_indices[label], count))
        else:
            raise ValueError(f"Not enough instances of class {label} to satisfy the requested count")

    # Create a subset of the dataset with the balanced indices
    balanced_subset = Subset(dataset, balanced_indices)
    return balanced_subset

# Function to apply t-SNE and visualize the results
def visualize_tsne(images, labels, class_names, model):

    # Evaluate the model and get features
    model.eval()
    with torch.no_grad():
        model_features = model(images)

    # Standardize the data before applying t-SNE
    scaler = StandardScaler()
    tsne = TSNE(n_components=2, random_state=42)

    # Standardize model features before applying t-SNE
    standardized_model_features = scaler.fit_transform(model_features.view(-1, model_features.size(-1)).cpu().numpy())

    # Apply t-SNE to model features
    reduced_features_model = tsne.fit_transform(standardized_model_features)

    # Plot the results
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        indices = labels == i
        plt.scatter(reduced_features_model[indices, 0], reduced_features_model[indices, 1], label=class_names[i])

    plt.title('t-SNE Visualization of Vanilla CL Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()

    return plt, model_features

def main(train_loader, valid_loader, valid_balanced_dataloader, seed):

    attn_model = initialize_model(seed)
    # Define loss function and optimizer

    if args['loss'] == 'HATCL_LOSS':
        cl_loss = HATCL_LOSS(temperature=args['temperature'])

    elif args['loss'] == 'LS_HATCL_LOSS':
        cl_loss = LS_HATCL_LOSS(temperature=args['temperature'])

    else:
        raise ValueError(f"Unsupported loss function: {args['loss']}")

    args['lr'] = float(args['lr'])

    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    else:
        optimizer = torch.optim.Adam(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.PATIENCE)



    # Wandb setup
    if config.WANDB:
        ds_name = os.path.realpath(ds_path).split('/')[-1]
        proj_name = 'Harth_cl_TRAIN_' + config.PROJ_NAME + ds_name
        wandb_logger = WandbLogger(project=proj_name)
        
        # Initialize Wandb
        wandb.init(project=proj_name)
        wandb.watch(attn_model, log='all', log_freq=100)

        # Update Wandb config
        wandb.config.update(ds_args)
        wandb.config.update(args)
        wandb.config.update({
            'Algorithm': 'VANILLA CONTRASTIVE LOSS',
            'Dataset': 'HUNT',
            'Train_DS_size': len(train_ds),
            'Batch_Size': args["batch_size"],
            'Epochs': args["epochs"],
            'Patience': config.PATIENCE,
            'Seed': seed

        })

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_model.to(device)

    # Training loop with best model saving
    best_val_loss = float('inf')  

    # Training loop with best model saving
    best_dbi = float('inf')    
    best_sc = -1.0 
    best_chi = 0

    # Training and validation loop
    num_epochs = args['epochs']
    mask_fraction = 0.3

    cluster_metrics = []
    for epoch in range(num_epochs):
        # Training phase
        attn_model.train()  # Set the model to training mode
        train_running_loss = 0.0

        for batch_idx, (time_series, labels) in enumerate(tqdm(train_loader)):
            time_series = time_series.to(device) 

            # Create a mask tensor with the same shape as the original tensor
            mask = torch.rand(time_series.shape) < mask_fraction
            mask = mask.to(device)

            # Apply the mask to the original tensor (e.g., setting masked values to -1)
            masked_tensor = torch.where(mask, torch.tensor(0), time_series).to(device)

            # Forward pass
            masked_features = attn_model(masked_tensor)

            # Flatten features to have dimensions [batch_size * sequence_length, feature dim]
            masked_features = masked_features.reshape(-1, masked_features.size(-1))
            time_series = time_series.reshape(-1, time_series.size(-1))
            
            linear_layer = nn.Linear(in_features=time_series.shape[-1], out_features=32)

            scaled_timeseries = linear_layer(time_series.float())

            # Compute training loss
            train_loss = F.mse_loss(scaled_timeseries, masked_features)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update training statistics
            train_running_loss += train_loss.item() * time_series.size(0)

        


        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}")

        # Log training loss to Wandb
        if config.WANDB and batch_idx % 10 == 0:
            wandb.log({'Train Loss': train_epoch_loss, 'Epoch': epoch})

        if epoch % config.VALID_INTERVAL == 0:
        
            attn_model.eval()  # Set the model to evaluation mode
            valid_running_loss = 0
            for batch_idx, (time_series, labels) in enumerate(tqdm(valid_loader)):
                time_series = time_series.to(device)

                # Create a mask tensor with the same shape as the original tensor
                mask = torch.rand(time_series.shape) < mask_fraction
                mask = mask.to(device)

                # Apply the mask to the original tensor (e.g., setting masked values to -1)
                masked_tensor = torch.where(mask, torch.tensor(0), time_series).to(device)

                # Forward pass
                masked_features = attn_model(masked_tensor)

                # Flatten features to have dimensions [batch_size * sequence_length, feature dim]
                masked_features = masked_features.reshape(-1, masked_features.size(-1))
                time_series = time_series.reshape(-1, time_series.size(-1))
                
                linear_layer = nn.Linear(in_features=time_series.shape[-1], out_features=32)

                scaled_timeseries = linear_layer(time_series.float())
                # Compute training loss
                
                valid_loss = F.mse_loss(scaled_timeseries, masked_features)

                # Update training statistics
                valid_running_loss += valid_loss.item() * time_series.size(0)

            scheduler.step(valid_running_loss)
            valid_epoch_loss = valid_running_loss / len(valid_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_epoch_loss:.4f}")

            for batch in valid_balanced_dataloader:
                images, labeli = batch
                images = images.view(-1, 1, images.shape[-1]).to(device)

            tsne_plot, time_features = visualize_tsne(images, labeli, class_dict, attn_model)

            # Calculate Davies-Bouldin Index
            db_index2 = davies_bouldin_score(time_features.cpu().detach().squeeze(), labeli)
            ch_index2 = calinski_harabasz_score(time_features.cpu().detach().squeeze(), labeli)
            slh_index2 = silhouette_score(time_features.cpu().detach().squeeze(), labeli)
            print(f"DB Index: {db_index2:.2f}, CH Index: {ch_index2:.2f}, SLH Index: {slh_index2:.2f}")

            cluster_metrics.append(0.33*((1/db_index2)+math.log(ch_index2 + 1) + 0.5*(slh_index2+1)))
            if config.WANDB:
                wandb.log({'Validation Loss': valid_epoch_loss, 'Epoch': epoch})
                wandb.log({'Davies-Bouldin Index Features': db_index2})
                wandb.log({'Calinski Harabasz Index Features': ch_index2})
                wandb.log({"Silhouette Index Features": slh_index2})
                wandb.log({"Joint cluster metrics": 0.33*((1/db_index2)+math.log(ch_index2 + 1) + 0.5*(slh_index2+1))})
                wandb.log({"t-SNE": wandb.Image(tsne_plot)})

            # Optionally save the model every config.SAVE_INTERVAL epochs
            if (epoch + 1) % config.SAVE_INTERVAL == 0 and db_index2 < best_dbi and slh_index2 > best_sc and ch_index2 > best_chi:
                best_dbi = db_index2 
                best_sc = slh_index2
                best_chi = ch_index2
                torch.save(attn_model.state_dict(), f'models/reconst_model_epoch_{epoch + 1}.pth')

    if config.WANDB:
        wandb.finish()
        
    print("Reconstruction Training and validation complete.")
    print("|")
    print(best_dbi, best_chi, best_sc)
    return cluster_metrics
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/harthconfig.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='data/harth')
    parser.add_argument('-s', '--seed_value', required=True, type=str,
                        help='seed value.', default='/')
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    seed = int(args.seed_value)
    # Log in to Wandb
    wandb.login(key=config.WANDB_KEY)

    # Set all seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            # Create the dataset
            if config.DATASET == 'STFT':
                dataset = src.data.get_dataset(
                        dataset_name=config.DATASET,
                        dataset_args=ds_args,
                        root_dir=ds_path,
                        num_classes=config.num_classes,
                        label_map=config.label_index,
                        replace_classes=config.replace_classes,
                        config_path=config.CONFIG_PATH,
                        name_label_map=config.class_name_label_map
                    )
                
            elif config.DATASET == 'ECG':
                dataset = STFTDataset(
                        data_path=ds_path,
                        n_fft = ds_args['n_fft'],
                        seq_length=ds_args['seq_length'],
                        class_to_exclude=ds_args['class_to_exclude'],
                        hop_length=ds_args['hop_length'],
                        win_length=ds_args['win_length'],
                        num_labels=ds_args['num_labels']
                    )

            elif config.DATASET == 'SLEEPEEG':
                dataset = SLEEPDataset(ds_path, seq_length=ds_args['seq_length'])

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")
            
            valid_split = config.VALID_SPLIT
            
            valid_amount = int(np.floor(len(dataset)*valid_split))
            train_amount = len(dataset) - valid_amount
            
            train_indices = list(range(train_amount))
            valid_indices = list(range(train_amount, train_amount + valid_amount))
            
            # Create subsets
            train_ds = Subset(dataset, train_indices)
            valid_ds = Subset(dataset, valid_indices)
            
            train_loader = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=args['batch_size'],
                # sampler=SequentialRandomSampler(train_ds, args['batch_size']),
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
            
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_ds,
                batch_size=args['batch_size'],
        #             sampler=SequentialRandomSampler(valid_ds, args['batch_size']),
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
    
    desired_count_per_class = config.class_count
    class_dict = config.class_dict

    flattened_data = FlattenedDataset(valid_ds)

    # Load balanced dataset

    balanced_dataset = load_balanced_dataset(flattened_data, desired_count_per_class)
    valid_balanced_dataloader = DataLoader(balanced_dataset, batch_size=config.display_batch, shuffle=False)


    main(train_loader, valid_loader, valid_balanced_dataloader, seed)