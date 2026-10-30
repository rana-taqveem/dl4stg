"""Fixed video splits and deterministic, epoch-specific window sampling."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from q2_core import make_window, sample_training_window


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_videos(root):
    from torchvision.datasets import MovingMNIST
    from torchvision.datasets.utils import check_integrity, download_url
    from pathlib import Path

    file = Path(root) / 'MovingMNIST/mnist_test_seq.npy'
    digest = 'be083ec986bfe91a449d63653c411eb2'

    if not check_integrity(str(file), digest):
        download_url('https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy',
                     root=str(file.parent), filename=file.name + '.download', md5=digest)
        file.with_name(file.name + '.download').replace(file)

    return MovingMNIST(root=str(root), split=None, download=True, transform=normalize_video)


def normalize_video(video):
    return video.to(torch.float32).div(255)


class Windows(Dataset):
    def __init__(self, videos, split='test', model='A', seed=0, epoch=0, speed=1):
        if len(videos) != 10_000:
            raise ValueError('Expected 10,000 complete videos.')

        ranges = {'train': range(8000), 'validation': range(8000, 9000),
                  'test': range(9000, 10000)}
        self.videos = Subset(videos, ranges[split])
        self.split, self.model, self.seed, self.epoch, self.speed = split, model, seed, epoch, speed

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]

        if self.split == 'train':
            # Independent per-video streams make resume/workers reproducible.
            rng = torch.Generator().manual_seed(self.seed * 1_000_000_000 + self.epoch * 10_000 + index)
            x, y, _ = sample_training_window(video, rng, self.model in ('C', 'D'), self.model == 'B')
        else:
            x, y = make_window(video, t=1, k=self.speed)

        return x, y


def collate_windows(batch):
    xs, ys = zip(*batch)
    target = ys[0].new_zeros(len(ys), 5, 1, 64, 64)
    mask = torch.zeros(len(ys), 5, dtype=torch.bool)

    for i, y in enumerate(ys):
        target[i, :len(y)] = y
        mask[i, :len(y)] = True

    return torch.stack(xs), target, mask


def loader(videos, device, split='test', model='A', seed=0, epoch=0, speed=1):
    return DataLoader(Windows(videos, split, model, seed, epoch, speed), batch_size=32,
                      shuffle=split == 'train', drop_last=False, num_workers=0,
                      pin_memory=device.type == 'cuda', collate_fn=collate_windows,
                      generator=torch.Generator().manual_seed(seed + epoch * 1_000_003))


def per_horizon_mse(prediction, target):
    return (prediction - target).square().mean(dim=(2, 3, 4))


def per_video_mse(prediction, target, mask):
    error = per_horizon_mse(prediction, target)
    return (error * mask).sum(1) / mask.sum(1)
