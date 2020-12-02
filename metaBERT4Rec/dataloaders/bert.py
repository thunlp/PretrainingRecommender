from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
import numpy as np
import torch
import torch.utils.data as data_utils
import random
import pickle

class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset, umap, smap):
        super().__init__(args, dataset, umap, smap)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.meta = pickle.load(open(args.meta, 'rb'))
        self.tmap = pickle.load(open(args.tmap, 'rb'))
        self.kg = args.kg
        self.mlm = args.mlm
        self.dire_size = args.dire_size
        self.acto_size = args.acto_size

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.meta, self.tmap, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng, ifkg=self.kg, ifmlm=self.mlm, dire_size=self.dire_size, acto_size=self.acto_size)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        '''
        t_ne = {}
        for ky in answers.keys():
            ans = answers[ky][0]
            for kki in range(len(self.smap)):
                if kki==ans:
                    continue
                if ky not in t_ne:
                    t_ne[ky] = [kki]
                else:
                    t_ne[ky].append(kki)
        '''
        dataset = BertEvalDataset(self.train, answers, self.meta, self.tmap, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, ifkg=self.kg, ifmlm=self.mlm, dire_size=self.dire_size, acto_size=self.acto_size)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, meta, tmap, max_len, mask_prob, mask_token, num_items, rng, ifkg=False, ifmlm=False, dire_size=-1, acto_size=-1):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.meta = meta
        self.tmap = tmap
        #self.rmap = rmap
        self.kg = ifkg
        self.mlm = ifmlm
        self.dire_size = dire_size
        self.acto_size = acto_size

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        tokens = []
        labels = []

        if self.mlm:
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items-1))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)
        else:
            if len(seq)<3:
                kk = len(seq)
            else:
                kk = random.randint(3, len(seq))
        
            tokens = seq[:kk-1]
            tokens.append(self.mask_token)
            labels = list(np.zeros(len(tokens)))
            labels[kk-1] = seq[kk-1]
        
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        lt = len(tokens)

        if self.kg:
            dire = list(np.ones(lt)*self.dire_size)
            ac1 = list(np.ones(lt)*self.acto_size)
            ac2 = list(np.ones(lt)*self.acto_size)
            ac3 = list(np.ones(lt)*self.acto_size)
            ac4 = list(np.ones(lt)*self.acto_size)
            for nm, sq in enumerate(tokens):
                if (sq==self.mask_token):# or (self.tmap[sq] not in self.meta):
                    continue
                if self.tmap[sq] not in self.meta:
                    continue
                tq = self.tmap[sq]
                dire[nm] = self.meta[tq][0][0]
                while len(self.meta[tq][1])<4:
                    self.meta[tq][1].append(self.acto_size)
                ac1[nm] = self.meta[tq][1][0]
                ac2[nm] = self.meta[tq][1][1]
                ac3[nm] = self.meta[tq][1][2]
                ac4[nm] = self.meta[tq][1][3]

            dire = dire[-self.max_len:]
            ac1 = ac1[-self.max_len:]
            ac2 = ac2[-self.max_len:]
            ac3 = ac3[-self.max_len:]
            ac4 = ac4[-self.max_len:]

            dire = [0] * mask_len + dire
            ac1 = [0] * mask_len + ac1
            ac2 = [0] * mask_len + ac2
            ac3 = [0] * mask_len + ac3
            ac4 = [0] * mask_len + ac4

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        
        if self.kg:
            return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(dire), torch.LongTensor(ac1), torch.LongTensor(ac2), torch.LongTensor(ac3), torch.LongTensor(ac4), user

        return torch.LongTensor(tokens), torch.LongTensor(labels), user


    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, meta, tmap, max_len, mask_token, negative_samples, ifkg=False, ifmlm=False, dire_size=-1, acto_size=-1):
        self.u2seq = u2seq
        self.meta = meta
        self.tmap = tmap
        #self.rmap = rmap
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
        #self.allc = np.load('/home1/private/zengzheni/BERT4Rec-VAE-Pytorch/Data/ml-1m/allc.npy')
        self.kg = ifkg
        self.mlm = ifmlm
        self.dire_size = dire_size
        self.acto_size = acto_size

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        #import pdb
        #pdb.set_trace()
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        lt = len(seq)
        padding_len = self.max_len - len(seq)
        if self.kg:
            dire = list(np.ones(lt)*self.dire_size)
            ac1 = list(np.ones(lt)*self.acto_size)
            ac2 = list(np.ones(lt)*self.acto_size)
            ac3 = list(np.ones(lt)*self.acto_size)
            ac4 = list(np.ones(lt)*self.acto_size)
            for nm, sq in enumerate(seq):
                if (sq==self.mask_token) or (self.tmap[sq] not in self.meta):
                    continue
                tq = self.tmap[sq]
                dire[nm] = self.meta[tq][0][0]
                while len(self.meta[tq][1])<4:
                    self.meta[tq][1].append(self.acto_size)
                ac1[nm] = self.meta[tq][1][0]
                ac2[nm] = self.meta[tq][1][1]
                ac3[nm] = self.meta[tq][1][2]
                ac4[nm] = self.meta[tq][1][3]
            dire = [0] * padding_len + dire
            ac1 = [0] * padding_len + ac1
            ac2 = [0] * padding_len + ac2
            ac3 = [0] * padding_len + ac3
            ac4 = [0] * padding_len + ac4
        seq = [0] * padding_len + seq
        if self.kg:
            return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(dire), torch.LongTensor(ac1), torch.LongTensor(ac2), torch.LongTensor(ac3), torch.LongTensor(ac4), user
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), user

