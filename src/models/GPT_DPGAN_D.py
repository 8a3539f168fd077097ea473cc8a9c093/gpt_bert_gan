import torch
import torch.nn.functional as F

import config as cfg
from models.generator import TransformerGenerator
from utils.data_loader import GenDataIter


class GPT_DPGAN_D(TransformerGenerator):
    def __init__(self, config):
        super(GPT_DPGAN_D, self).__init__(config)

    def getReward(self, samples):
        """
        Get word-level reward and sentence-level reward of samples.
        """
        batch_size, _ = samples.size()
        inp, target = GenDataIter.prepare(samples, cfg.CUDA) # [batch_size, max_seq_len], [batch_size, max_seq_len]
        inp = inp.transpose(1, 0)       # [max_seq_len, batch_size]
        target = target.transpose(1, 0) # [max_seq_len, batch_size]

        
        dummy_tgt = torch.ones(self.max_seq_len, batch_size, dtype=torch.int)
        if self.gpu:
            dummy_tgt = dummy_tgt.cuda()

        pred = self.forward(target, inp)

        word_reward = F.nll_loss(pred, target.contiguous().view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)

        #print(f"word reward len: {word_reward.size()} {word_reward}")
        #print(f"word reward: {word_reward}")

        #print(f"sentence reward len: {sentence_reward.size()}")
        #print(f"sentence reward:{sentence_reward}")

        return word_reward, sentence_reward