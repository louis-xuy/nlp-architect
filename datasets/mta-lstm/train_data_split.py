# -*- coding: utf-8 -*-

"""
Created on 2018/10/16 下午4:47

@author: xujiang@baixing.com

"""

with open('zhihu.txt', 'r', encoding='utf-8') as inf, open('train.txt', 'w', encoding='utf-8') as outf:
    for index, line in enumerate(inf):
        if index>1000:
            break
        outf.write(line)
