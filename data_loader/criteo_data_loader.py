from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader
from dataset import CriteoDataset


class CriteoDataLoader(BaseDataLoader):
    """
    Criteo data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = CriteoDataset(self.data_dir, read_part=True, sample_num=100000)
        #img,label = self.dataset[0]
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def get_field_dims(self):
        return self.dataset.get_field_dims()