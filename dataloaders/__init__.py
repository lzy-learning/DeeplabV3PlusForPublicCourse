from dataloaders.datasets import camvid
from torch.utils.data import DataLoader
from config import config


def make_data_loader(**kwargs):
    if config.dataset == 'camvid':
        train_set = camvid.CamVidDataset(config.args, split='train')
        val_set = camvid.CamVidDataset(config.args, split='val')
        num_class = train_set.NUM_CLASS
        train_loader = DataLoader(train_set, batch_size=config.args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=config.args.batch_size, shuffle=True, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
