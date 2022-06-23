# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : dpgan_instructor.py
# @Time         : Created at 2019/12/21
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import json
import nltk

import torch
import torch.optim as optim
import transformers
from transformers import GPT2Model, GPT2Tokenizer, get_linear_schedule_with_warmup

import config as cfg
from instructor.real_data.instructor import SelfAttentionInstructor
from models.BERT_sentiment import BERT_sentiment
from models.GPT_2 import GPT_2
from utils import helpers
from utils.bp_encoder import get_encoder
from utils.gpt2_data_loader import GenDataIter
from utils.text_process import write_tokens, load_dict, tensor_to_tokens, tokens_to_tensor, text_process
from torchvision import models
from torchsummary import summary
from torch import nn
from utils import text_process

import visual.training_plots


class GPT_BERT_DPGAN(SelfAttentionInstructor):
    def __init__(self, opt):
        super(GPT_BERT_DPGAN, self).__init__(opt)

        # generator, discriminator
        self.gen = GPT_2()
        self.dis = BERT_sentiment(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                  cfg.padding_idx, gpu=cfg.CUDA)

        # Load weights from huggingface GPT_2 transformer class
        pretrained_model = GPT2Model.from_pretrained("gpt2")
        pretrained_model.cuda()

        self.gen = helpers.load_weight(self.gen, pretrained_model.state_dict())
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        # self.gen_adv_opt = optim.SGD(self.gen.parameters(), lr=cfg.gen_lr)
        #self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        # most default parameters for AdamW should be good
        # otw. look at https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
        self.gen_adv_opt = optim.AdamW(self.gen.parameters(), lr=cfg.gen_lr, weight_decay=0)
        #self.gen_adv_opt = transformers.AdamW(self.gen.parameters(), lr=cfg.gen_lr)
        #self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        # we want to warmup for one epoch(for coco dataset we need 10 mini-batches to have
        # the whole dataset and here we test 20 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.gen_adv_opt, num_warmup_steps=10,  num_training_steps=10 * cfg.ADV_train_epoch
        )

        # Tokenizer for the pretrained gpt2
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.bpe = get_encoder()

        # load dictionary
        self.log.info(f"Loading {cfg.dataset} dataset")
        with open('utils/encoder.json', 'r') as f:
            encoder = json.load(f)
            decoder = {v: k for k, v in encoder.items()}
            self.word2idx_dict, self.idx2word_dict = encoder, decoder

        # recorded info for ploting
        self.rating_bins = []

        # Dataloader
        try:
            self.train_data = GenDataIter(cfg.train_data)
            self.test_data = GenDataIter(cfg.test_data, if_test_data=True)
        except:
            pass

        try:
            self.train_data_list = [GenDataIter(cfg.cat_train_data.format(i)) for i in range(cfg.k_label)]
            self.test_data_list = [GenDataIter(cfg.cat_test_data.format(i), if_test_data=True) for i in
                                   range(cfg.k_label)]
            self.clas_data_list = [GenDataIter(cfg.cat_test_data.format(str(i)), if_test_data=True) for i in
                                   range(cfg.k_label)]

            self.train_samples_list = [self.train_data_list[i].target for i in range(cfg.k_label)]
            self.clas_samples_list = [self.clas_data_list[i].target for i in range(cfg.k_label)]
        except:
            pass
        self.word2idx_dict_old, self.idx2word_dict_old = load_dict(cfg.dataset)

    def init_model(self):
        """
        Overwrites the init_model() in instructor.py
        """
        if cfg.CUDA:
            self.gen = self.gen.cuda()
            # self.dis = self.dis.cuda()

    def _run(self):
        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))
        if cfg.run_validation:
            for adv_epoch in range(cfg.ADV_train_epoch):
                self.test_model_on_dataset(adv_epoch)

                # first epoch is for original gpt-2 and second is for trained gpt-2
                if adv_epoch == 2:
                    break
        else :
            for adv_epoch in range(cfg.ADV_train_epoch):
                self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
                self.sig.update()
                if self.sig.adv_sig:
                    rating_bin = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                    self.log.info("RATING_BINS: EPOCH{}".format(adv_epoch))
                    self.log.info("number of samples classified as negative: " + str(rating_bin[0]))
                    self.log.info("number of samples classified as neutral: " + str(rating_bin[1]))
                    self.log.info("number of samples classified as positive: " + str(rating_bin[2]))
                    if (adv_epoch + 1) % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                        if cfg.if_save and not cfg.if_test:
                            self._save('ADV', adv_epoch)
                    if adv_epoch % 2 == 0:
                        self.rating_bins.append(rating_bin)
                    if adv_epoch == 12:
                        # visual.training_plots.plot_ratings(self.rating_bins)
                        visual.training_plots.plot_ratings_12(self.rating_bins)
                else:
                    self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                    break

    def _test(self):
        print('>>> Begin test...')
        self._run()
        pass

    def test_model_on_dataset(self, adv_epoch):
        """
        Compare GPT-2 without finetuning to GPT-2 with finetuning on a particular dataset.
        """
        if adv_epoch == 0:
            rating_bin = self.sample_sentiment()
        if adv_epoch == 1:
            self.log.info('Load fine_tuned nice generator: {}'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path, map_location='cuda:{}'.format(cfg.device)))
            rating_bin = self.sample_sentiment()
        if adv_epoch == 2:
            visual.training_plots.plot_ratings_compared(self.rating_bins)
            return
        self.log.info("RATING_BINS: EPOCH{}".format(adv_epoch))
        self.log.info("number of samples classified as negative: " + str(rating_bin[0]))
        self.log.info("number of samples classified as neutral: " + str(rating_bin[1]))
        self.log.info("number of samples classified as positive: " + str(rating_bin[2]))
        if adv_epoch % 1 == 0:
            self.rating_bins.append(rating_bin)

    def sample_sentiment(self):
        """
        Function to be called before training to get an estimate of the sentiments of the generated sentences.
        """

        training_bin = [0 for i in range(3)]
        data_loader = self.train_data.loader
        for count, data in enumerate(data_loader):
            inp = data['input']
            if cfg.CUDA:
                inp = inp.cuda()

            # generate one sample from context
            gen_samples, gen_sample_log_prob = self.gen.sample_teacher_forcing(inp)

            # give reward to the generated sample
            sentence_sentiment = self.dis.getReward(gen_samples, training_bin)

            # display example of generation/reward for the first batch
            if count == 0:
                samples = gen_samples.tolist()
                samples = [[self.bpe.decode(sample)] for sample in samples]
                self.log.info("SAMPLES: ")
                self.log.info(samples)
                self.log.info("SENTIMENT_SCORE: {}".format(sentence_sentiment))

        return training_bin

    def adv_train_generator(self, g_step):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        total_g_loss = 0

        training_bin = [0 for i in range(3)]
        data_loader = self.train_data.loader
        self.gen_adv_opt.zero_grad()
        for count, data in enumerate(data_loader):
            inp = data['input']
            if cfg.CUDA:
                inp = inp.cuda()

            # generate one sample from context
            gen_samples, gen_sample_log_prob = self.gen.sample_teacher_forcing(inp)

            # give reward to the generated sample
            sentence_sentiment = self.dis.getReward(gen_samples, training_bin)

            # display example of generation/reward for the first batch
            if count == 0:
                samples = gen_samples.tolist()
                samples = [[self.bpe.decode(sample)] for sample in samples]
                self.log.info("SAMPLES: ")
                self.log.info(samples)
                self.log.info("SENTIMENT_SCORE: {}".format(sentence_sentiment))

            if cfg.CUDA:
                sentence_sentiment = sentence_sentiment.cuda()

            # attribute this reward to each token and compute loss
            sentence_sentiment = sentence_sentiment * gen_sample_log_prob
            word_sentiments = sentence_sentiment.repeat(cfg.max_seq_len, 1)
            word_sentiments = torch.transpose(word_sentiments, 0, 1)
            target_sentiments = torch.full_like(word_sentiments, 1)
            if cfg.CUDA:
                word_sentiments = word_sentiments.cuda()
                target_sentiments = target_sentiments.cuda()
            #loss = nn.MSELoss()
            loss = nn.L1Loss()
            # loss = nn.BCEWithLogitsLoss()
            # loss = nn.CrossEntropyLoss()
            adv_loss = loss(word_sentiments, target_sentiments)

            if cfg.CUDA:
                adv_loss = adv_loss.cuda()
                # self.optimize(self.gen_adv_opt, adv_loss, self.gen)

            # accumulate the gradients
            adv_loss.backward()
            total_g_loss += adv_loss.item()

            # accumulate the gradient and update weights only when enough accumulated
            # this allow us to have batch size = eg 16 and effectively 128
            if count % 32 == 0:
                self.optimize(self.gen_adv_opt, adv_loss, self.gen)
                self.scheduler.step()
        # ===Test===
        self.log.info(
            '[ADV-GEN]: g_loss = %.4f' % (total_g_loss / (g_step * cfg.batch_size)))

        return training_bin

    def eval_dis(self, model, pos_val, neg_val):
        _, pos_reward = model.getReward(pos_val)
        _, neg_reward = model.getReward(neg_val)
        return torch.mean(pos_reward), torch.mean(neg_reward)

    def cal_metrics(self, fmt_str=False):
        """
        Overwrites cal_metrics from BasicInstructor, because we need to use a specific
        tokenizer for the pretrained gpt2
        """
        with torch.no_grad():
            # Prepare data for evaluation
            if cfg.show_probs:
                show_probs = True
            else:
                show_probs = False
            eval_samples, _ = self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                                       batch_size=cfg.samples_num, temperature=0.7, top_k=40,
                                                       sample_pos2=show_probs)
            gen_data = GenDataIter(eval_samples)
            # gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            eval_samples = eval_samples.tolist()
            gen_tokens = [[self.bpe.decode(eval_sample)] for eval_sample in eval_samples]
            # gen_tokens_s = tensor_to_tokens(self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
            #                                        batch_size=200, temperature=0.7, top_k=40), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader)
            self.nll_div.reset(self.gen, gen_data.loader)
            # self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            #self.ppl.reset(gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]

    def _save(self, phase, epoch):
        """Overwrites _save in instructor to add gpt2 tokenizer"""
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples, _ = self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                              batch_size=50, temperature=0.7, top_k=40)
        samples = samples.tolist()
        # samples = [[self.tokenizer.decode(sample)] for sample in samples]
        samples = [[self.bpe.decode(sample)] for sample in samples]
        write_tokens(save_sample_path, samples)

    def create_fake_dataset(self):
        """
        Function used to create a dataset of fake generated text using GPT-2 with
        a particular dataset as context for generation.
        """
        data_loader = self.train_data.loader
        fake_sentences = []
        for count, data in enumerate(data_loader):
            inp = data['input']
            if cfg.CUDA:
                inp = inp.cuda()

            # generate one sample from context
            samples = self.gen.sample_teacher_forcing(inp)
            samples = samples[0].tolist()
            samples = [self.bpe.decode(sample) for sample in samples]
            for sample in samples:
                sample_tokenized = nltk.word_tokenize(sample)
                #sample_tokenized = sample
                fake_sentences.append(sample_tokenized)
        text_process.write_tokens(cfg.save_samples_root + 'fake_dataset.txt', fake_sentences)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        # loss.backward(retain_graph=retain_graph)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        opt.zero_grad()
