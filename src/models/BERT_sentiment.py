# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
from torch import nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from utils.bp_encoder import get_encoder
from utils.data_loader import GenDataIter
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


class BERT_sentiment(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(BERT_sentiment, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'roberta'
        self.bpe = get_encoder()
        self.model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.sentiment = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.softmax = nn.Softmax(dim=1)
    def score_token(self, score, label):
        if label == 'POSITIVE':
            return score
        else:
            return 1.0 - score

    def sentiment_to_score(self, scores):
        """
        Works with score with 3 classes (neg, neutral, pos)
        """
        rewards = []
        for score in list(scores):
            neutral_score = score[1].item() * 0.5
            positive_score = score[2].item() * 1
            rewards.append(neutral_score + positive_score)
        return rewards
    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    def getBinIndex(self, value):
        bin_values = [0.0, 0.5, 1.0]
        bin_distances = np.array([abs(bin_value - value) for bin_value in bin_values])
        bin_index = np.argmin(bin_distances)
        return bin_index


    def getReward(self, samples, training_bin, one_sample=False, pos_or_neg_sample=None):
        """
        Get word-level reward and sentence-level reward of samples.
        """

        """
        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)
        """
        with torch.no_grad():
            
            if one_sample:
                samples = self.bpe.decode(samples.tolist())
            else:
                samples = samples.tolist()
                samples = [self.preprocess(self.bpe.decode(sample)) for sample in samples]
            # TODO: would be better to use the input as a tensor to be
            # able to use the gpu
            #print("samples")
            #print(samples)
            encoded_input = self.tokenizer(samples, return_tensors='pt', padding=True)
            """
            if cfg.CUDA:
                encoded_input['input_ids'] = encoded_input['input_ids'].cuda()
                encoded_input['attention_mask'] = encoded_input['attention_mask'].cuda()
            """
            output = self.sentiment(**encoded_input)
            #print("output[0]")
            #print(output[0])
            score_pt_large = self.softmax(output[0])
            #print("score_pt_large")
            #print(score_pt_large)
            sentences_reward = self.sentiment_to_score(score_pt_large)
            #print("sentences_reward")
            #print(sentences_reward)
            sentence_sentiment = torch.tensor(sentences_reward, requires_grad=False)
            for sentiment in sentences_reward:
                training_bin[self.getBinIndex(sentiment)] += 1
        """
        print("SAMPLES_BERT")
        print(samples)
        print("SENTIMENTS")
        print(sentiments)
        """

        """
        label_map = {'NEGATIVE': 0.0, 'POSITIVE': 1.0}
        sentence_sentiment = torch.tensor([self.score_token(sentiment['score'], sentiment['label']) for sentiment in
                                         sentiments], requires_grad=False)
        #sentence_sentiment = sentence_rewards.view(1, len(sentence_rewards))
        # maybe better to give rewards for only this length and not cfg.max_seqlen
        #word_rewards = [sentence_rewards for i in range(len(samples[0]))]
        for sentiment in sentiments:
            training_bin[int(label_map[sentiment['label']])] += 1
        #print("SENTENCE_SENTIMENT")
        #print(sentence_sentiment)
        """
        return sentence_sentiment

