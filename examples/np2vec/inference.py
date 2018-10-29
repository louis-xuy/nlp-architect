# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import random
import argparse
import logging
import sys

from nlp_architect.models.np2vec import NP2vec
from nlp_architect.utils.io import validate_existing_filepath, check_size

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

ouf = open('result.csv', 'w', encoding="utf-8")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--np2vec_model_file',
        default='conll2000.train.model',
        help='path to the file with the np2vec model to load.',
        type=validate_existing_filepath)
    arg_parser.add_argument(
        '--binary',
        help='boolean indicating whether the model to load has been stored in binary '
        'format.',
        action='store_true')
    arg_parser.add_argument(
        '--word_ngrams',
        default=0,
        type=int,
        choices=[0, 1],
        help='If 0, the model to load stores word information. If 1, the model to load stores '
        'subword (ngrams) information; note that subword information is relevant only to '
        'fasttext models.')
    arg_parser.add_argument(
        '--mark_char',
        default='_',
        type=str,
        action=check_size(1, 2),
        help='special character that marks word separator and NP suffix.')
    arg_parser.add_argument(
        '--np',
        default='Intel Corp.',
        type=str,
        action=check_size(min_size=1),
        required=True,
        help='NP to print its word vector.')

    args = arg_parser.parse_args()

    np2vec_model = NP2vec.load(
        args.np2vec_model_file,
        binary=args.binary,
        word_ngrams=args.word_ngrams)

    # print("word vector for the NP \'" + args.np + "\':", np2vec_model[args.mark_char.join(
    #    args.np.split()) + args.mark_char])
    
    
    arr = [
    '全新',
    '斗山',
    '60 - 7',
    '挖掘机',
    '野马',
    'T70',
    '1.8',
    'T',
    '无级',
    '升级版进取',
    '大众',
    'POLO',
    'Cross',
    '北京',
    '旗岭',
    '箱货',
    '4.3',
    '米',
    '大箱',
    '解放',
    '半挂车',
    '国6',
    '国5',
    '国4',
    '国六',
    '国五',
    '国四',
    '奥迪A4',
    '豪华型',
    '起亚K2',
    'K2',
    '自动',
    '抵押车',
    '仓库',
    '批发',
    '一手',
    '安全',
    '正规',
    '招中介',
    '常年',
    '收售',
    '二手',
    '农机',
    '原装',
    '进口',
    '日立240 - 3',
    '日立',
    '挖掘机',
    '单位',
    '低价',
    '处理',
    '7',
    '吨',
    '10',
    '吨',
    '叉车',
    '转让',
    '东风',
    '天龙',
    '前四后四',
    '载货车',
    '220',
    '马',
    '马力',
    '9.6',
    '米厢',
    '华菱',
    '高配',
    '后八轮',
    '便宜卖'
    ]
    for keyword in arr:
        try:
            result = np2vec_model.similar_by_word(keyword, 20)
            ouf.write("{}, {}\n".format(keyword, '--'))
        except Exception as e:
            print(e)
            continue
        for _ in range(10):
            slice = random.sample(result, 4)
            ouf.write("{}, {}\n".format("", ''.join([t[0] for t in slice])))
    ouf.close()
