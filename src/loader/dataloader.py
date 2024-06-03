import torch
import numpy as np




class SequentialRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size


    def __iter__(self):
        

        indices = list(range(len(self.data_source)))
        
        remaining = len(indices) % self.batch_size
        if remaining > 0:
            indices = indices[:-remaining]
        final_indices = np.reshape(indices, (-1, self.batch_size))

        # Shuffle the batches
        np.random.shuffle(final_indices)

        # Flatten the list of batches to get the final order of indices
        final_indices = [idx for batch in final_indices for idx in batch]
        
        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)