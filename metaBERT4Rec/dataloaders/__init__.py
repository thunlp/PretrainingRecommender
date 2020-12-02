from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader
import pickle

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    umap = pickle.load(open(args.umap, 'rb'))
    smap = pickle.load(open(args.smap, 'rb'))
    dataloader = dataloader(args, dataset, umap, smap)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
