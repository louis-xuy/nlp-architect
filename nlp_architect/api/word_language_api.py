# -*- coding: utf-8 -*-

"""
Created on 2018/10/10 上午11:21

@author: xujiang@baixing.com

"""

import os
import pickle
import jieba
import numpy as np
from os import path, makedirs, sys
import tensorflow as tf
from nlp_architect.utils.text import Vocabulary
from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.gen_char_rnn import CharRNN


class WordLanguageApi(AbstractApi):
    dir = path.dirname(path.realpath(__file__))
    pretrained_model = path.join(dir, 'gen-pretrained')
    pretrained_model_info = path.join(dir, 'gen-pretrained', 'model_info.dat')

    def __init__(self, ner_model=None, prompt=True):
        self.model = None
        self.vocabulary = None
        self.model_path = WordLanguageApi.pretrained_model
        self.model_info_path = WordLanguageApi.pretrained_model_info
        
    def load_model(self):
        with open(self.model_info_path, 'rb') as fp:
            self.model_info = pickle.load(fp)
        print (self.model_info)
            
        
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)

        self.vocabulary = Vocabulary(vocab_path=os.path.join(self.pretrained_model, 'vocabs.pkl'))
    
        self.model = CharRNN(self.vocabulary.max,
                                sampling=True,
                                n_neurons=self.model_info['n_neurons'],
                                n_layers=self.model_info['n_layers'],
                                embedding=self.model_info['embedding'],
                                embedding_size=self.model_info['embedding_size']
                        )
    
        self.model.load(checkpoint_path)

    def pretty_print(self, text):
        spans = []
        ret = {}
        ret['doc_text'] =text
        ret['annotation_set'] = []
        ret['spans'] = spans
        ret['title'] = 'None'
        return {"doc": ret, 'type': 'high_level'}
    
    def inference(self, doc):
        start = np.array([self.vocabulary.word_id(word) for word in jieba.cut(doc)])
        print (start)
        print (self.model_info['vocab_size'])
        arr = self.model.predict(50, start, self.model_info['vocab_size'])
        text = ''.join([self.vocabulary.id_to_word(wid) for wid in arr])
        text = text.replace('EB','<\n>')
        return self.pretty_print(text)