# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 11/03/2017 9:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
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
# ------------------------------------------------------------------------
<<<<<<< HEAD

import os
import copy
import jieba
import pickle
import numpy as np
import collections
import tensorflow as tf
from nlp_architect.utils.text import Vocabulary
from nlp_architect.models.gen_char_rnn import CharRNN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('batch_size', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('n_neurons', 128, 'size of hidden state(neurons) of lstm')
tf.flags.DEFINE_integer('n_layers', 3, 'number of lstm layers')
tf.flags.DEFINE_boolean('embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer('n_iterations', 10000, 'number of iterations to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('num_words', 5000, 'number of words in the vocabulary')

FLAGS.input_file = 'datasets/gen_data/gongchengche_2018_10_08.csv'

model_info_path = 'nlp_architect/api/gen-pretrained/model_info.dat'


def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


def main(_):
    start_token = 'B'
    end_token = 'E'
    vocabs = Vocabulary(2)
    model_path = 'nlp_architect/api/gen-pretrained/'
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    datas = []
    with open(FLAGS.input_file, encoding='utf-8') as f:
        texts = f.readlines()
        for text in texts:
            content = text.replace(' ', '')
            print(content)
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                    start_token in content or end_token in content:
                continue
            content = start_token + content + end_token
            datas.append(content)
    all_words = [word for text in datas for word in jieba.cut(text)]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    #count_pairs = count_pairs[:FLAGS.num_words]
    words, _ = zip(*count_pairs)
    for word in words:
        vocabs.add(word)
    print(vocabs.max)
    with open(os.path.join(model_path, 'vocabs.pkl'), 'wb') as f:
        pickle.dump(vocabs.vocab, f)
    arr = np.array([vocabs.word_id(word) for data in datas for word in jieba.cut(data)])
    batch = batch_generator(arr, FLAGS.batch_size, FLAGS.num_steps)
    model = CharRNN(vocabs.max,
                    batch_size=FLAGS.batch_size,
                    num_steps=FLAGS.num_steps,
                    n_neurons=FLAGS.n_neurons,
                    n_layers=FLAGS.n_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    embedding=FLAGS.embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.fit(batch,
                FLAGS.n_iterations,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )
    with open(model_info_path, 'wb') as fp:
        info = {
            'vocab_size': vocabs.max,
            'batch_size': FLAGS.batch_size,
            'num_steps': FLAGS.num_steps,
            'n_neurons': FLAGS.n_neurons,
            'n_layers': FLAGS.n_layers,
            'learning_rate': FLAGS.learning_rate,
            'train_keep_prob': FLAGS.train_keep_prob,
            'embedding': FLAGS.embedding,
            'embedding_size': FLAGS.embedding_size
        }
        pickle.dump(info, fp)


if __name__ == '__main__':
    tf.app.run()
=======
import os
import numpy as np
import tensorflow as tf
from gen_char_rnn import rnn_model
from data_process import process_datas, generate_batch

tf.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.flags.DEFINE_string('model_dir', os.path.abspath('../../nlp_architect/api/gen-pretrained'), 'model save path.')
tf.flags.DEFINE_string('file_path', os.path.abspath('../../datasets/gen_data/gongchengche_2018_10_08.csv'), 'file name of datas.')
tf.flags.DEFINE_string('model_prefix', 'gcc', 'model save prefix.')
tf.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.flags.FLAGS


def run_training():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    datas_vector, word_to_int, vocabularies = process_datas(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, datas_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=3, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(datas_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
>>>>>>> e522a9c8f7e87ff12ae737526caa085c7647cfc4
