from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from .ev import evaluate
import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.kg = args.kg

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        if self.kg:
            seqs, labels, dire, ac1, ac2, ac3, ac4, user = batch
            logits = self.model(seqs, user, dire, ac1, ac2, ac3, ac4)  # B x T x V
        else:
            seqs, labels, user = batch
            logits = self.model(seqs, user)
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, test=False):
        if self.kg:
            seqs, candidates, labels, dire, ac1, ac2, ac3, ac4, user = batch
            scores = self.model(seqs, user, dire, ac1, ac2, ac3, ac4)  # B x T x V
        else:
            seqs, candidates, labels, user = batch
            scores = self.model(seqs, user)
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        if test:
            metrics = evaluate(scores, labels, self.metric_ks)
        else:
            metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
