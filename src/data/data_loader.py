import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

def get_data_loader(data_dir: str, 
                    image_size: tuple = (3, 128, 128),
                    batch_size: int = 32, 
                    num_workers: int = 4):
    # Create the dataset
    dataset = dset.ImageFolder(root=data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers,)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers,)
    

    return train_loader, val_loader



class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image
