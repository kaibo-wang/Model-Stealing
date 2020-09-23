import inspect
from itertools import product
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataset_configuration import *


class TransformedDataset(Dataset):

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        self._len = images.size()[0]

    def __getitem__(self, item):
        image, label = self.images[item].squeeze(0), self.labels[item]
        image = Image.fromarray(image.numpy(), mode='L')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self._len


class DatasetLoader:

    def __init__(self, dataset_config: DataSetConfiguration,
                 transform=None,
                 dataset_name='',
                 train_name='', val_name='', test_name='',
                 train_image_name='', train_label_name='',
                 val_image_name='', var_label_name='',
                 test_image_name='', test_label_name=''):
        arg_dict = inspect.getargvalues(inspect.currentframe()).locals
        del arg_dict['self']
        self.__dict__.update(arg_dict)
        if not self.dataset_name:
            self.dataset_name = self.dataset_config.name

    def prepare_data(self):
        data_dic = {}
        load_dic = {i: j for i, j in self.__dict__.items() if isinstance(j, str) and j}
        keys = set('_'.join(i) for i in product(('train', 'test', 'val'), ('image', 'label')))
        if not keys < self.__dict__.keys():
            for arg_name, arg in load_dic.items():
                path = self.dataset_config.dataset / (arg + '.pt')
                if path.exists():
                    data_dic[arg_name[:-5]] = torch.load(path)
            for key in keys:
                self.__dict__[key] = self.get_data(data_dic, key)

    @staticmethod
    def _get_value(dic: dict, key: str, legal_type):
        value = dic[key]
        assert isinstance(value, legal_type)
        return value

    def get_data(self, data_dic: dict, name: str):
        prefix, postfix = name.split('_', 2)
        if name in data_dic.keys():
            return self._get_value(data_dic, name, torch.Tensor)
        elif prefix in data_dic.keys():
            dic = self._get_value(data_dic, prefix, dict)
            if name in dic.keys():
                return self._get_value(dic, name, torch.Tensor)
            elif postfix in dic.keys():
                return self._get_value(dic, postfix, torch.Tensor)
        elif 'dataset' in data_dic.keys():
            dic = self._get_value(data_dic, 'dataset', dict)
            if name in dic.keys():
                return self._get_value(dic, name, torch.Tensor)
        return

    def get_dataset(self, prefix: str, transform=None):
        transform = transform if transform else self.transform
        images = self.__dict__[prefix + '_image']
        labels = self.__dict__[prefix + '_label']
        if isinstance(images, torch.Tensor) and isinstance(labels, torch.Tensor):
            images.detach_()
            labels.detach_()
            assert images.size()[0] == labels.size()[0]
            assert images.size()[1] == self.dataset_config.channel \
                   and images.size()[2] == self.dataset_config.height \
                   and images.size()[3] == self.dataset_config.width
            if transform:
                return TransformedDataset(images, labels, transform)
            else:
                return TensorDataset(images, labels)
        elif isinstance(labels, tuple):
            for label in labels:
                label.detach_()
            return TensorDataset(images, *labels)
