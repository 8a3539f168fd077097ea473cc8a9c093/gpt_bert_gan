import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

import config as cfg
from models.generator import TransformerGenerator
from utils.bp_encoder import get_encoder
from utils.text_process import cut_eot_token


class GPT_2(TransformerGenerator):
    def __init__(self):
        self.config = cfg.GPT2Config(vocab_size_or_config_json_file=50257,
                                     n_positions=1024,
                                     n_ctx=1024,
                                     n_embd=768,
                                     n_layer=12,
                                     n_head=12,
                                     layer_norm_epsilon=1e-5,
                                     initializer_range=0.02)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.bpe = get_encoder()
        super(GPT_2, self).__init__(self.config)

    """
    def sample_sequence(self, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                        device='cuda', sample=True, sample_pos2=False):
        # TODO: should I assume that the input is already tokenized or maybe I can tokenize here
    """

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
        """
        samples = torch.zeros(batch_size, cfg.max_seq_len - 1).long()
        for i in range(batch_size):
            context = inp[i]

            out = cut_eot_token(context)
            if out is not None:
                context = out
            sampled, log_probs = self.sample_sequence(cfg.max_seq_len - 1, context=context, start_token=None,
                                                    batch_size=10, temperature=0.7,
                                                     top_k=40)
            #sampled = sampled[:, len(context):]
            sampled = sampled[:, :cfg.max_seq_len - 1]
            samples[i, :] = sampled.view(len(sampled[0]))
        """
        # TODO: this line is problematic but necessary

        context = [cut_eot_token(inpi, 8)[:8] for inpi in inp]
        context = torch.stack(context)

        if cfg.CUDA:
            context = context.cuda()
        sampled, log_probs = self.sample_sequence(cfg.max_seq_len - 1, context=context, start_token=None,
                                                  batch_size=cfg.batch_size, temperature=0.7,
                                                  top_k=40, skip_context_init=True)

        return sampled, log_probs

    def sample_sequence(self, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                        device='cuda', sample=True, sample_pos2=False, skip_context_init=False):
        """
        Overwrites sample_sequence in generator.py to add the tokenizer for pretrained gpt2
        """

        if not skip_context_init:
            if start_token is None:
                assert context is not None, 'Specify exactly one of start_token and context!'
                context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
                # context = context.clone().detach().float().requires_grad_(True).unsqueeze(0).repeat(batch_size, 1)
            else:
                assert context is None, 'Specify exactly one of start_token and context!'
                context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
                # device =...
        prev = context
        output = context
        past = None
        if cfg.CUDA:
            prev, output = prev.cuda(), output.cuda()
        # with torch.no_grad():
        for i in range(length):
            logits, past = self(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = self.top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample_pos2:
                if i == 1:
                    print("10 most probable token at position 2: ")
                    sample_prob = log_probs[0]
                    prob_top_pred, sample_top_pred = torch.topk(sample_prob, 10)
                    tokens = [self.idx2word_dict[str(i)] for i in sample_top_pred.tolist()]
                    print(prob_top_pred)
                    print(tokens)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            # print("output{}".format(i))
            # print(output)

            output = torch.cat((output, prev), dim=1)
        return output, torch.sum(log_probs) / cfg.batch_size
