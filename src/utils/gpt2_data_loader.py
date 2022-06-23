import json
import random

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import config as cfg
from utils.bp_encoder import get_encoder

from utils.text_process import get_tokenlized, tokens_to_tensor, load_dict, load_test_dict, get_raw_text


class gpt2_data_loader(Dataset):
    def __init__(self):
        self.tokens = None
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    @staticmethod
    def tokenize_function(examples):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
        )

    @staticmethod
    def gpt2_tensor_to_token(tokenizer, tokens):
        text = ''.join([tokenizer.decode(token) for token in tokens])
        return text


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GenDataIter:
    def __init__(self, samples, if_test_data=False, shuffle=None):
        self.bpe = get_encoder()
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        with open('utils/encoder.json', 'r') as f:
            encoder = json.load(f)
            decoder = {v: k for k, v in encoder.items()}
        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = encoder, decoder
        if if_test_data:  # used for the classifier
            self.word2idx_dict, self.idx2word_dict = encoder, decoder

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)
        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target = self.prepare(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        elif isinstance(samples, str):  # filename
            inp, target = self.load_data(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        else:
            all_data = None
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        #inp[:, 0] = cfg.start_letter
        inp[:, :] = target[:, :cfg.max_seq_len]

        #print(f"dataloader inp: {inp[0][:]}")
        #print(f"dataloader target: {target[0][:]}")

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        texts = get_raw_text(filename)
        # samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        samples_index = [self.bpe.encode(text) for text in texts]
        samples_index = torch.LongTensor(samples_index)
        return self.prepare(samples_index)


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

    def __read_data__(self, pos_samples, neg_samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(pos_samples, neg_samples)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def prepare(self, pos_samples, neg_samples, gpu=False):
        """Build inp and target"""
        inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()  # !!!need .detach()
        target = torch.ones(inp.size(0)).long()
        target[pos_samples.size(0):] = 0

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target
