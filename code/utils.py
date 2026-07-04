import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in /"
                f"sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


import numpy as np
import torch


class EarlyStopping:
    """早停机制实现类"""

    def __init__(self, patience=7, min_delta=0, mode='max', verbose=True):
        """
        参数:
            patience (int): 在多少个epoch内验证指标没有提升就停止训练
            min_delta (float): 最小变化阈值，只有超过这个值才认为是提升
            mode (str): 'min' 或 'max'，表示监控的指标是越小越好还是越大越好
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch, val_score, model, path):
        """
        每个epoch调用一次，判断是否需要早停

        参数:
            epoch (int): 当前epoch
            val_score (float): 验证集上的指标值
            model (nn.Module): 模型实例
            path (str): 模型保存路径
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_score, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered. Best epoch: {self.best_epoch}')
        else:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(val_score, model, path)

    def save_checkpoint(self, val_score, model, path):
        """保存最佳模型"""
        if self.verbose:
            print(f'Validation score improved ({self.val_score:.6f} --> {val_score:.6f}). Saving model ...')
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': model.state_dict(),
            'val_score': val_score,
        }, path)
        self.val_score = val_score
