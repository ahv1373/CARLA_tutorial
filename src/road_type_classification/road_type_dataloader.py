from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class RoadTypeDataset(Dataset):
    def __init__(self, data_dir: str):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Better to compute the mean and std of the dataset or use pre-computed values from ImageNet
            # ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (These are the standard values)
        ])
        self.dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

def get_dataloader(data_dir: str, batch_size: int = 4, num_workers: int = 2, shuffle: bool = True,
                   pin_memory: bool = True, persistent_workers: bool = True, prefetch_factor: int = 2):
    dataset = RoadTypeDataset(data_dir)
    '''
    pin_memory=True: This allows the data loader to copy Tensors into CUDA pinned memory before returning them.
    This can save some time when transferring data to the GPU.
    persistent_workers=True: This allows the workers to persist between data loader iterations, which can improve
    performance if data loading is a bottleneck.
    prefetch_factor=2: This allows the data loader to prefetch additional data from the dataset, which can improve
    performance by reducing I/O wait time.
    '''
    dataloader_ = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return dataloader_


if __name__ == '__main__':
    # Test the dataloader
    import os
    import torch
    from tqdm import tqdm
    dataloader = get_dataloader(os.path.join(os.path.dirname(__file__), '..' , 'data', 'train'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for inputs, labels in tqdm(dataloader, desc="Iterating dataloader"):
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs.shape, labels)