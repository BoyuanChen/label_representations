import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path


# Dataset for CIFAR10 and CIFAR100.
class CIFARDataset(Dataset):
    # Initializes the dataset for the given datatype, label, and number of classes.
    def __init__(self, data_dir, datatype, label, num_classes, label_root=None):
        assert datatype in ("train", "test", "valid")
        assert ("category" in label) or (
            label
            in (
                "speech",
                "uniform",
                "shuffle",
                "composite",
                "random",
                "lowdim",
                "bert",
                "bert_filter",
                "glove",
            )
        )
        assert num_classes in (10, 100)
        self.label = label
        self.num_classes = num_classes

        if datatype == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        train_or_valid = datatype in ("train", "valid")
        if num_classes == 10:
            self.dataset = datasets.CIFAR10(
                root=data_dir,
                train=train_or_valid,
                download=True,
                transform=transform,
            )
        else:  # num_classes == 100:
            self.dataset = datasets.CIFAR100(
                root=data_dir,
                train=train_or_valid,
                download=True,
                transform=transform,
            )

        # Loads high-dimensional targets.
        if "category" not in label:
            assert label_root is not None
            self.mels = np.load(
                Path(label_root)
                / "cifar{}_{}.npy".format(self.num_classes, self.label)
            )

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)
        if "category" in self.label:
            return (img, target)
        else:  # high-dimensional data
            return (img, self.mels[target])

    def __len__(self):
        return self.dataset.__len__()


# Loads train dataset where the data sequence is seeded by the given seed.
# Note: data_level in percentage.
def get_train_loader(
    data_dir,
    label,
    num_classes,
    num_workers,
    batch_size,
    seq_seed,
    data_level=100,
    label_root=None,
):
    trainset = CIFARDataset(
        data_dir=data_dir, datatype="train", label=label, num_classes=num_classes, label_root=label_root
    )
    indices = list(range(len(trainset)))
    np.random.seed(seq_seed)
    np.random.shuffle(indices)
    split = 5000
    train_len = int(len(indices) * data_level / 100)
    train_idx = indices[split : split + train_len]
    trainsampler = SubsetRandomSampler(train_idx)
    return torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers
    )


# Loads validation dataset where the data sequence is seeded by the given seed.
def get_valid_loader(
    data_dir, label, num_classes, num_workers, batch_size, seq_seed, label_root=None
):
    validset = CIFARDataset(
        data_dir=data_dir, datatype="valid", label=label, num_classes=num_classes, label_root=label_root
    )
    indices = list(range(len(validset)))
    np.random.seed(seq_seed)
    np.random.shuffle(indices)
    split = 5000
    valid_idx = indices[:split]
    validsampler = SubsetRandomSampler(valid_idx)
    return torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers
    )


# Loads test dataset without data shuffling.
def get_test_loader(data_dir, label, num_classes, num_workers, batch_size, label_root=None):
    testset = CIFARDataset(
        data_dir=data_dir, datatype="test", label=label, num_classes=num_classes, label_root=label_root
    )
    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
