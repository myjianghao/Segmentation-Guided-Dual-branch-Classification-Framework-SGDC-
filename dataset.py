from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from typing import List, Optional, Sequence, Union, Any, Callable
from PIL import Image
import random
import json
import torch
import numpy as np


class Test_Dataset(Dataset):
    def __init__(self, file_list, label_dict, transform, mask, is_regression):
        self.file_list = file_list
        self.label_dict = label_dict
        self.transform = transform
        self.mask = mask
        self.is_regression = is_regression

    def __getitem__(self, index):
        if self.mask:
            X = Image.open(self.file_list[index])
            if (X.mode != 'RGB'):
                X = X.convert("RGB")
            X_rsize = self.transform(X)
            X_m = Image.open(self.file_list[index][:-4] + '_m.png')
            X_mrsize = self.transform(X_m)
            # hard mask
            data_m = (torch.sum(X_mrsize, dim=0) > 0) * 1.0
            data = torch.cat((data_m.unsqueeze(0), X_rsize), dim=0)
            if self.is_regression:
                labels = torch.tensor(self.label_dict[self.file_list[index]], dtype=torch.float32)
            else:
                labels = self.label_dict[self.file_list[index]]
        else:
            X = Image.open(self.file_list[index])
            if (X.mode != 'RGB'):
                X = X.convert("RGB")
            X_rsize = self.transform(X)
            # data_m = torch.zeros((X_rsize.shape[1], X_rsize.shape[2]))
            # data = torch.cat((data_m.unsqueeze(0), X_rsize), dim=0)
            data = X_rsize
            if self.is_regression:
                labels = torch.tensor(self.label_dict[self.file_list[index]], dtype=torch.float32)
            else:
                labels = self.label_dict[self.file_list[index]]
        filename = self.file_list[index]
        return data, labels, filename

    def __len__(self):
        return len(self.file_list)


class MyDataset(Dataset):
    def __init__(self, file_list, label_dict, transform, mask, is_regression):
        self.file_list = file_list
        self.label_dict = label_dict
        self.transform = transform
        self.is_regression = is_regression
        self.mask = mask
        self.count = 0

    def __getitem__(self, index):
        if self.mask:
            X = Image.open(self.file_list[index])
            if (X.mode != 'RGB'):
                X = X.convert("RGB")
            X_rsize = self.transform(X)
            """X_m = Image.open(self.file_list[index][:-4] + '_m.png')
            X_mrsize = self.transform(X_m)
            # hard mask
            data_m = (torch.sum(X_mrsize, dim=0) > 0) * 1.0
            data = torch.cat((data_m.unsqueeze(0), X_rsize), dim=0)"""
            data = X_rsize
            if self.is_regression:
                labels = torch.tensor(self.label_dict[self.file_list[index]], dtype=torch.float32)
            else:
                labels = self.label_dict[self.file_list[index]]
            """decay_rate = 0.95
            decay_element = decay_rate ** (self.count / 100)
            # soft mask
            data_m[data_m == 0] = decay_element
            data = X_rsize * data_m
            labels = self.label_dict[self.file_list[index][:-6] + '.JPG']
            self.count += 1"""
        else:
            X = Image.open(self.file_list[index])
            if (X.mode != 'RGB'):
                X = X.convert("RGB")
            X_rsize = self.transform(X)
            # data_m = torch.zeros((X_rsize.shape[1], X_rsize.shape[2]))
            # data = torch.cat((data_m.unsqueeze(0), X_rsize), dim=0)
            data = X_rsize
            if self.is_regression:
                labels = torch.tensor(self.label_dict[self.file_list[index]], dtype=torch.float32)
            else:
                labels = self.label_dict[self.file_list[index]]
        return data, labels

    def __len__(self):
        return len(self.file_list)


class ResnetDataset(LightningDataModule):
    def __init__(
            self,
            file_dict: dict,
            mask: bool = False,
            is_regression: bool = False,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            val_size: int = 50,
            test_size: int = 50,
            patch_size: Union[int, Sequence[int]] = (660, 900),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.file_dict = file_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.patch_size = tuple(list(map(int, patch_size.split(','))))
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mask = mask
        self.is_regression = is_regression

    def setup(self, stage: Optional[str] = None) -> None:
        train_transformer = transforms.Compose([transforms.Resize(self.patch_size),
                                                transforms.ToTensor()])
        val_transformer = transforms.Compose([transforms.Resize(self.patch_size),
                                              transforms.ToTensor()])
        file_list = list(self.file_dict.keys())
        random.seed(17)
        random.shuffle(file_list)
        """train_file_list = file_list[0:-self.val_size - self.test_size]
        train_file_array = np.array(train_file_list, dtype='<U30')
        if self.mask:
            for mask_file in self.mask_dict["mask_file"]:
                train_file_array[train_file_array == (mask_file[:-6] + '.JPG')] = mask_file
                # train_file_list.append(mask_file)
        train_file_list = train_file_array.tolist()"""
        self.train_dataset = MyDataset(file_list[0:-self.val_size - self.test_size], self.file_dict, train_transformer,
                                       self.mask, self.is_regression)
        """self.val_dataset = MyDataset(file_list[-self.val_size - self.test_size:- self.test_size], self.file_dict,
                                     val_transformer)"""
        self.val_dataset = MyDataset(file_list[-self.val_size - self.test_size:], self.file_dict, val_transformer,
                                     self.mask, self.is_regression)
        self.test_dataset = Test_Dataset(file_list[- self.test_size:], self.file_dict, val_transformer,
                                         self.mask, self.is_regression)
        # self.test_dataset = Test_Dataset(file_list, file_dict, val_transformer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )


class Segmentation_Dataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        X = Image.open(self.file_list[index])
        if (X.mode != 'RGB'):
            X = X.convert("RGB")
        data = self.transform(X)
        resize = transforms.Resize((224, 256))
        X_m = Image.open(self.file_list[index][:-4] + '_m.png')
        X_mrsize = self.transform(X_m)
        # hard mask
        data_m = (torch.sum(X_mrsize, dim=0) > 0) * 1.0
        labels = data_m.unsqueeze(0)
        return resize(data), resize(labels)

    def __len__(self):
        return len(self.file_list)


class UnetDataset(LightningDataModule):
    def __init__(
            self,
            file_dict: dict,
            mask: bool = False,
            is_regression: bool = False,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            val_size: int = 50,
            test_size: int = 50,
            patch_size: Union[int, Sequence[int]] = (660, 900),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.file_dict = file_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.patch_size = tuple(list(map(int, patch_size.split(','))))
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mask = mask
        self.is_regression = is_regression

    def setup(self, stage: Optional[str] = None) -> None:
        train_transformer = transforms.Compose([transforms.CenterCrop(self.patch_size),
                                                transforms.ToTensor()])
        val_transformer = transforms.Compose([transforms.CenterCrop(self.patch_size),
                                              transforms.ToTensor()])
        file_list = list(self.file_dict.keys())
        random.seed(17)
        random.shuffle(file_list)
        """train_file_list = file_list[0:-self.val_size - self.test_size]
        train_file_array = np.array(train_file_list, dtype='<U30')
        if self.mask:
            for mask_file in self.mask_dict["mask_file"]:
                train_file_array[train_file_array == (mask_file[:-6] + '.JPG')] = mask_file
                # train_file_list.append(mask_file)
        train_file_list = train_file_array.tolist()"""
        self.train_dataset = Segmentation_Dataset(file_list[0:-self.val_size - self.test_size], train_transformer)
        """self.val_dataset = MyDataset(file_list[-self.val_size - self.test_size:- self.test_size], self.file_dict,
                                     val_transformer)"""
        self.val_dataset = Segmentation_Dataset(file_list[-self.val_size - self.test_size:], val_transformer)
        self.test_dataset = Segmentation_Dataset(file_list[- self.test_size:], val_transformer)
        # self.test_dataset = Test_Dataset(file_list, file_dict, val_transformer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )


class Seg_ClassDataset(Dataset):
    def __init__(self, file_list, label_dict, transform, is_regression):
        self.file_list = file_list
        self.label_dict = label_dict
        self.transform = transform
        self.is_regression = is_regression

    def __getitem__(self, index):
        X = Image.open(self.file_list[index])
        if (X.mode != 'RGB'):
            X = X.convert("RGB")
        data = self.transform(X)
        X_m = Image.open(self.file_list[index][:-4] + '_m.png')
        X_mrsize = self.transform(X_m)
        resize = transforms.Resize((224, 256))
        # hard mask
        data_m = (torch.sum(X_mrsize, dim=0) > 0) * 1.0
        s_labels = data_m.unsqueeze(0)
        if self.is_regression:
            c_labels = torch.tensor(self.label_dict[self.file_list[index]], dtype=torch.float32)
        else:
            c_labels = self.label_dict[self.file_list[index]]
        return data, resize(s_labels), c_labels

    def __len__(self):
        return len(self.file_list)


class Unet_ResnetDataset(LightningDataModule):
    def __init__(
            self,
            file_dict: dict,
            mask: bool = False,
            is_regression: bool = False,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            val_size: int = 50,
            test_size: int = 50,
            patch_size: Union[int, Sequence[int]] = (660, 900),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.file_dict = file_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.patch_size = tuple(list(map(int, patch_size.split(','))))
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mask = mask
        self.is_regression = is_regression

    def setup(self, stage: Optional[str] = None) -> None:
        train_transformer = transforms.Compose([transforms.Resize(self.patch_size),
                                                transforms.ToTensor()])
        val_transformer = transforms.Compose([transforms.Resize(self.patch_size),
                                              transforms.ToTensor()])
        file_list = list(self.file_dict.keys())
        random.seed(17)
        random.shuffle(file_list)
        """train_file_list = file_list[0:-self.val_size - self.test_size]
        train_file_array = np.array(train_file_list, dtype='<U30')
        if self.mask:
            for mask_file in self.mask_dict["mask_file"]:
                train_file_array[train_file_array == (mask_file[:-6] + '.JPG')] = mask_file
                # train_file_list.append(mask_file)
        train_file_list = train_file_array.tolist()"""
        self.train_dataset = Seg_ClassDataset(file_list[0:-self.val_size - self.test_size], self.file_dict,
                                              train_transformer,
                                              self.is_regression)
        """self.val_dataset = MyDataset(file_list[-self.val_size - self.test_size:- self.test_size], self.file_dict,
                                     val_transformer)"""
        self.val_dataset = Seg_ClassDataset(file_list[-self.val_size - self.test_size:], self.file_dict,
                                            val_transformer,
                                            self.is_regression)
        self.test_dataset = Seg_ClassDataset(file_list[- self.test_size:], self.file_dict, val_transformer,
                                             self.is_regression)
        # self.test_dataset = Test_Dataset(file_list, file_dict, val_transformer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )


from torchvision.transforms import InterpolationMode


def multi_transforms(img_size, train):
    if train:
        trans = [[transforms.CenterCrop(img_size), transforms.Resize(img_size), transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.CenterCrop(img_size),
                  transforms.RandomResizedCrop(img_size, scale=(1, 1), interpolation=InterpolationMode.BICUBIC),
                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.CenterCrop(img_size), transforms.Resize(img_size), transforms.RandomHorizontalFlip(p=1),
                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.CenterCrop(img_size), transforms.Resize(img_size), transforms.RandomVerticalFlip(p=1),
                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.CenterCrop(img_size), transforms.Resize(img_size), transforms.RandomRotation(270),
                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.CenterCrop(img_size), transforms.Resize(img_size), transforms.RandomRotation(180),
                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.CenterCrop(img_size), transforms.Resize(img_size), transforms.RandomRotation(90),
                  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]]
    else:
        trans = [transforms.CenterCrop(img_size), transforms.Resize(img_size),
                 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return trans


class load_CSFT(Dataset):
    def __init__(self, file_list, label_dict, is_regression, train, transform, transform2, transform4):
        self.file_list = file_list
        self.label_dict = label_dict
        self.is_regression = is_regression
        self.train = train
        self.transform = transform
        self.transform2 = transform2
        self.transform4 = transform4

    def __len__(self):
        if self.train:
            return len(self.file_list) * len(self.transform)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.file_list[idx // len(self.transform)]
            img = Image.open(img_path)
            if (img.mode != 'RGB'):
                img = img.convert("RGB")
            transformed = transforms.Compose(self.transform[idx % len(self.transform)])
            transformed2 = transforms.Compose(self.transform2[idx % len(self.transform)])
            transformed4 = transforms.Compose(self.transform4[idx % len(self.transform)])
        else:
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform)
            transformed2 = transforms.Compose(self.transform2)
            transformed4 = transforms.Compose(self.transform4)
        img_transformed = transformed(img)
        img_transformed2 = transformed2(img)
        img_transformed4 = transformed4(img)
        if self.is_regression:
            labels = torch.tensor(self.label_dict[self.file_list[idx // len(self.transform)]], dtype=torch.float32)
        else:
            labels = self.label_dict[self.file_list[idx // len(self.transform)]]
        return img_transformed, img_transformed2, img_transformed4, labels


class CSFTDataset(LightningDataModule):
    def __init__(
            self,
            file_dict: dict,
            mask: bool = False,
            is_regression: bool = False,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            val_size: int = 50,
            test_size: int = 50,
            patch_size: Union[int, Sequence[int]] = (660, 900),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.file_dict = file_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.patch_size = tuple(list(map(int, patch_size.split(','))))
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mask = mask
        self.is_regression = is_regression

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = multi_transforms(self.patch_size, True)
        train_transforms2 = multi_transforms((self.patch_size[0] // 2, self.patch_size[1] // 2), True)
        train_transforms4 = multi_transforms((self.patch_size[0] // 4, self.patch_size[1] // 4), True)
        test_transforms = multi_transforms(self.patch_size, False)
        test_transforms2 = multi_transforms((self.patch_size[0] // 2, self.patch_size[1] // 2), False)
        test_transforms4 = multi_transforms((self.patch_size[0] // 4, self.patch_size[1] // 4), False)
        file_list = list(self.file_dict.keys())
        random.seed(17)
        random.shuffle(file_list)

        self.train_dataset = load_CSFT(file_list[0:-self.val_size - self.test_size], self.file_dict, self.is_regression,
                                       True, train_transforms, train_transforms2, train_transforms4)
        self.val_dataset = load_CSFT(file_list[-self.val_size - self.test_size:], self.file_dict, self.is_regression,
                                     False, test_transforms, test_transforms2, test_transforms4)
        self.test_dataset = load_CSFT(file_list[- self.test_size:], self.file_dict, self.is_regression, False,
                                      test_transforms, test_transforms2, test_transforms4)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )
