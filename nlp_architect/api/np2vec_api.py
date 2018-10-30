# -*- coding: utf-8 -*-

"""
Created on 2018/10/30 1:48 PM

@author: xujiang@baixing.com

"""

from os import path
import random
from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.np2vec import NP2vec


class Np2VecApi(AbstractApi):
    dir = path.dirname(path.realpath(__file__))
    model_path = path.join(dir, 'np2vec-pretrained', 'Tencent_AILab_ChineseEmbedding.txt')

    
    def __init__(self):
        self.model = None
        self.model_path = Np2VecApi.model_path
    
    def load_model(self):
        self.model = NP2vec.load(
            self.model_path,
            binary=False,
            word_ngrams=0)
    
    def pretty_print(self, text):
        spans = []
        ret = {}
        ret['doc_text'] =text
        ret['annotation_set'] = []
        ret['spans'] = spans
        ret['title'] = 'None'
        return {"doc": ret, 'type': 'high_level'}
    
    def inference(self, doc):
        keyword = doc.strip()
        result_arr = self.model.similar_by_word(keyword, 20)
        results = []
        for _ in range(10):
            slice = random.sample(result_arr, 3)
            results.append(keyword+"".join([t[0] for t in slice]))
        text = "\n".join(results)
        return self.pretty_print(text)
        
        
        
        