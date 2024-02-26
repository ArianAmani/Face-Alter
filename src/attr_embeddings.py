from typing import Union
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.module import ConvolutionalVariationalAutoencoder
from utils.conf import image_size
from data.data_loader import ImageDataset


def create_attr_embeddings(attr_csv: Union[str, pd.DataFrame],
                           model_path: str,
                           dataset_path: str,
                           target_attrs: list,
                           device: str = 'cuda') -> pd.DataFrame:

    checkpoint = torch.load(model_path)
    model = ConvolutionalVariationalAutoencoder().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if type(attr_csv) == str:
        attr_csv = pd.read_csv(attr_csv)
    elif type(attr_csv) == pd.DataFrame:
        pass
    else:
        raise TypeError('attr_csv must be a string path or a pandas DataFrame')

    attr_embeddings = {}
    for attr in target_attrs:
        has_attr = attr_csv[attr_csv[attr] == 1]['image_id'].values
        no_attr = attr_csv[attr_csv[attr] == -1]['image_id'].values

        has_attr_paths = 'img_align_celeba/img_align_celeba/' + has_attr
        no_attr_paths = 'img_align_celeba/img_align_celeba/' + no_attr
        
        # Transformations
        transformations = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # Create dataset and dataloader
        has_attr_dataset = ImageDataset(has_attr_paths, transformations)
        has_attr_dataloader = DataLoader(has_attr_dataset, batch_size=32, shuffle=False)

        no_attr_dataset = ImageDataset(no_attr_paths, transformations)
        no_attr_dataloader = DataLoader(no_attr_dataset, batch_size=32, shuffle=False)

        # Get embeddings
        has_attr_embs = []
        for _, data in enumerate(has_attr_dataloader):
            data = data.to(device)
            z = model.get_embedding(data)
            has_attr_embs.append(z)
            
        has_attr_embedding = torch.cat(has_attr_embs, dim=0).mean(dim=0)
        
        no_attr_embs = []
        for _, data in enumerate(no_attr_dataloader):
            data = data.to(device)
            z = model.get_embedding(data)
            no_attr_embs.append(z)
            
        no_attr_embedding = torch.cat(no_attr_embs, dim=0).mean(dim=0)
        
        attr_embeddings[attr] = has_attr_embedding - no_attr_embedding
        
    