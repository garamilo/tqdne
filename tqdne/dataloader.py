import logging
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdne.dataset import Dataset


def get_train_and_val_loader(config, num_workers, batchsize, cond=False):
    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=cond, split="train"
    )
    val_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=cond, split="validation"
    )

    # === revision: adding weighted random sampler on cases where mag-dist population is really skewed to one side
    sampler = None
    shuffle = True

    if cond:
        logging.info("Building WeightedRandomSampler to balance classes...")
        
        dist = train_dataset.file["hypocentral_distance"][:]
        mag = train_dataset.file["magnitude"][:]
        
        labels = ((np.digitize(dist, config.dist_bins) - 1) * (len(config.mag_bins) - 1) + np.digitize(mag, config.mag_bins) - 1)
        
        train_labels = labels[train_dataset.indices]

        class_counts = np.bincount(train_labels)
        weights = 1.0 / np.where(class_counts > 0, class_counts, 1)
        sample_weights = [weights[l] for l in train_labels]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)   
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        prefetch_factor=1,
        persistent_workers=True,
        pin_memory=True
    )

    return train_loader, val_loader
