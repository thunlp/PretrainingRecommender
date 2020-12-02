from torch import nn as nn
import torch

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as
import pickle

class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        #fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        self.meta = pickle.load(open(args.meta, 'rb'))

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)
        self.ue = BERTEmbedding(vocab_size=6040, embed_size=int(self.hidden/4), max_len=max_len, dropout=dropout)
        if args.kg:
            self.dir = BERTEmbedding(vocab_size=args.dire_size+1, embed_size=int(self.hidden/8), max_len=max_len, dropout=dropout)
            self.act = BERTEmbedding(vocab_size=args.acto_size+1, embed_size=int(self.hidden/8), max_len=max_len, dropout=dropout)
        # multi-layers transformer blocks, deep network
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(int(hidden*7/4), heads, hidden * 4, dropout) for _ in range(n_layers)])
        else:
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(int(hidden*5/4), heads, hidden * 4, dropout) for _ in range(n_layers)])


    def forward(self, x, user, dire, ac1, ac2, ac3, ac4):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        user = user.unsqueeze(1)
        user = user.expand(user.shape[0], x.shape[1])
        u = self.ue(user)
        if dire is None:
            f_in = torch.cat((x, u), axis=2)
        else:
            dire = self.dir(dire)
            ac1 = self.act(ac1)
            ac2 = self.act(ac2)
            ac3 = self.act(ac3)
            f_in = torch.cat((x,dire), axis=2)
            f_in = torch.cat((f_in,ac1), axis=2)
            f_in = torch.cat((f_in,ac2), axis=2)
            f_in = torch.cat((f_in,ac3), axis=2)
            f_in = torch.cat((f_in,u), axis=2)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            f_in = transformer.forward(f_in, mask)

        return f_in

    def init_weights(self):
        pass
