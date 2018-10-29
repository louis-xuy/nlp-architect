# -*- coding: utf-8 -*-

"""
Created on 2018/10/27 11:33 AM

@author: xujiang@baixing.com

"""
import jieba
import pickle
from os import path

from keras.preprocessing.sequence import pad_sequences

from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.supervised_sentiment import SentimentLSTM

class SentimentClassifyApi(AbstractApi):
    """
    Sentiment model API
    """
    dir = path.dirname(path.realpath(__file__))
    pretrained_model = path.join(dir, 'sentiment-pretrained', 'model.h5')
    pretrained_model_info = path.join(dir, 'sentiment-pretrained', 'model_info.dat')

    def __init__(self, sentiment_model=None, prompt=True):
        self.model = None
        self.model_info = None
        self.model_path = SentimentClassifyApi.pretrained_model
        self.model_info_path = SentimentClassifyApi.pretrained_model_info
        
    def pretty_print(self, data):
        pass
        
    def load_model(self):
        with open(self.model_info_path, 'rb') as fp:
            self.model_info = pickle.load(fp)
            self.tokenizer = self.model_info['tokenizer']
            self.model = SentimentLSTM()
            self.model.build(
                self.model_info['max_fature'],
                self.model_info['num_of_labels'],
                self.model_info['input_length'],
                self.model_info['embed_dim'],
                self.model_info['lstm_out'],
                self.model_info['dropout'])
            self.model.load(self.model_path)
    
    def process_text(self, text):
        text = text.strip()
        text = self.tokenizer.texts_to_sequences([' '.join(jieba.cut(text))])
        text = pad_sequences(text, maxlen=300)
        return text
    
    def inference(self, doc):
        text = self.process_text(doc)
        result = self.model.predict(text)
        print(result)
        