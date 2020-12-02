from .base import BaseModel
from .bert_modules.bert import BERT
import torch
import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        #self.ue = nn.Embedding(6040, 50)
        if args.kg:
            self.out = nn.Linear(int(self.bert.hidden*7/4), args.num_items + 1)
        else:
            self.out = nn.Linear(int(self.bert.hidden*5/4), args.num_items + 1)
    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x, user, dire=None, ac1=None, ac2=None, ac3=None, ac4=None):
        f_in = self.bert(x, user, dire, ac1, ac2, ac3, ac4)
        return self.out(f_in)
