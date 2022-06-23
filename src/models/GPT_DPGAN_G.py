import torch
import torch.nn.functional as F

from models.generator import TransformerGenerator
import config as cfg


class GPT_DPGAN_G(TransformerGenerator):
    def __init__(self, config):
        super(GPT_DPGAN_G, self).__init__(config)

    def sample_teacher_forcing(self, inp):
        """
        Generating samples from the real data via teacher forcing
        :param inp: batch_size * seq_len
        :param target: batch_size * seq_len
        :return
            samples: batch_size * seq_len
            log_prob: batch_size * seq_len  (log probabilities)
        """
        batch_size, _ = inp.size()

        logits, past = self(inp)
        pred = F.softmax(logits, dim=-1)
        samples = self.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                                     batch_size=batch_size, temperature=0.7,
                                                     top_k=1, sample=False)
        log_prob = F.nll_loss(pred.view(-1, cfg.GPT2Config().vocab_size), samples.view(-1), reduction='none').view(batch_size, -1)
        # samples = torch.multinomial(torch.exp(log_prob), 1)

        return samples, log_prob

