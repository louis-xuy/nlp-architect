# -*- coding: utf-8 -*-

"""
Created on 2018/10/19 上午11:12

@author: xujiang@baixing.com

"""


from keras_contrib.utils import save_load_utils


class BaseModel(object):
    
    def save(self, path):
        """
        Save model to path

        Args:
            path (str): path to save model weights
        """
        save_load_utils.save_all_weights(self.model, path)

    def load(self, path):
        """
        Load model weights

        Args:
            path (str): path to load model from
        """
        save_load_utils.load_all_weights(self.model, path, include_optimizer=False)
        print('testing model:', self.model.predict(np.zeros((1, 30))))