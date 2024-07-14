import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from osgeo import gdal

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
    '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


class WHU_OHS_Dataset(Dataset):
    def __init__(self, image_file_list, label_file_list):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list

    def sample_stat(self):
        """Statistics of samples of each class in the dataset."""
        sample_per_class = torch.zeros([24])
        for label_file in self.label_file_list:
            label = gdal.Open(label_file, gdal.GA_ReadOnly)
            label = label.ReadAsArray()
            count = np.bincount(label.ravel(), minlength=25)
            count = count[1:25]
            count = torch.tensor(count)
            sample_per_class += count

        return sample_per_class

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image = image_dataset.ReadAsArray()
        label = label_dataset.ReadAsArray()
        image = torch.tensor(image, dtype=torch.float) / 10000.0
        label = torch.tensor(label, dtype=torch.float) - 1.0
        return image, label, name.split(".")[0]

def get_dataset_loader(txt_file_path, data_path, batch_size, shuffle):
    image_list = []
    label_list = []

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            image_path = os.path.join(data_path, line[0] + '.tif')
            label_path = os.path.join(data_path.replace('image', 'label'), line[0] + '.tif')

            assert os.path.exists(label_path), f"{label_path} does not exist!"
            assert os.path.exists(image_path), f"{image_path} does not exist!"

            image_list.append(image_path)
            label_list.append(label_path)

    assert len(image_list) == len(label_list), "The number of images and labels must be equal!"

    dataset = WHU_OHS_Dataset(
        image_file_list=image_list,
        label_file_list=label_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )
    return loader


def load_data(args, mode='tr'):
    assert mode in ['tr', 'val', 'ts'], "Invalid mode. Mode should be either 'tr', 'val' or 'ts'."
    data_path = os.path.join(args.data_root, mode, 'image')
    is_shuffle = True if mode == 'tr' else False
    txt_file_path = os.path.join("txts/", mode + ".txt")
    loader = get_dataset_loader(
        txt_file_path,
        data_path,
        args.batch_size,
        is_shuffle,
    )
    return loader
