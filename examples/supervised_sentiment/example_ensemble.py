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

"""
This example uses the Amazon reviews though additional datasets can easily be substituted.
It only requires text and a sentiment label
It then takes the dataset and trains two models (again can be expanded)
The labels for the test data is then predicted.
The same train and test data is used for both models

The ensembler takes the two prediction matrixes and weights (as defined by model accuracy)
and determines the final prediction matrix.

Finally, the full classification report is displayed.

A similar pipeline could be utilized to train models on a dataset, predict on a second dataset
and aquire a list of final predictions
"""

import os
import pickle
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nlp_architect.models.supervised_sentiment import SentimentLSTM
from nlp_architect.utils.io import validate_existing_filepath, check_size

def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_fatures', type=int, default=2000)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--lstm_out', type=int, default=140)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--file_path', type=str,
                        default='/Users/louis/WorkSpace/PycharmProjects/nlp-architect/datasets/supervised_sentiment/',
                        help='file_path where the files to parse are located')
    parser.add_argument('--data_type', type=str, default='amazon',
                        choices=['amazon'],
                        help='dataset source')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for both models', action=check_size(1, 20000))
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to external word embedding model file')
    parser.add_argument('--model_path', type=str, default='model.h5',
                        help='Path for saving model weights')
    parser.add_argument('--model_info_path', type=str, default='model_info.dat',
                        help='Path for saving model topology')
    args_in = parser.parse_args()
    return args_in


# def ensemble_models(Y_train, X_train_, Y_test, X_test_, args):
#     # split, train, test
#     # data.process()
#     dense_out = 2
#     # split for all models
#     #X_train_, X_test_, Y_train, Y_test = train_test_split(data.text, data.labels,
#     #                                                      test_size=0.20, random_state=42)
#
#
#
#     # Prep data for the LSTM model
#     tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#     tokenizer.fit_on_texts(X_train_)
#     X_train = tokenizer.texts_to_sequences(X_train_)
#     X_train = pad_sequences(X_train, maxlen=max_len)
#     X_test = tokenizer.texts_to_sequences(X_test_)
#     X_test = pad_sequences(X_test, maxlen=max_len)
#
#     # Y_train = to_categorical(Y_train)
#     # Y_test = to_categorical(Y_test)
#     dic = {
#             '__label__1':0,
#             '__label__5':1
#     }
#     Y_train = list(map(lambda x:dic[x], Y_train))
#     Y_train = to_categorical(Y_train)
#     Y_test = list(map(lambda x:dic[x], Y_test))
#     Y_test = to_categorical(Y_test)
#
#
#
#     # Train the LSTM model
#     lstm_model = SentimentLSTM()
#     lstm_model.fit(max_fatures, dense_out, X_train.shape[1], embed_dim, lstm_out)
#     # lstm_model = simple_lstm(max_fatures, dense_out, X_train.shape[1], embed_dim, lstm_out)
#     lstm_model.fit(X_train, Y_train, epochs=args.epochs, batch_size=batch_size,
#                                 verbose=1, validation=(X_test, Y_test))
#     lstm_model.save()
    # lstm_acc = model_hist.history['acc'][-1]
    # print("LSTM model accuracy ", lstm_acc)

    # And make predictions using the LSTM model
    # lstm_predictions = lstm_model.predict(X_test)

    # # Now prep data for the one-hot CNN model
    # X_train_cnn = np.asarray([to_one_hot(x) for x in X_train_])
    # X_test_cnn = np.asarray([to_one_hot(x) for x in X_test_])
    #
    # # And train the one-hot CNN classifier
    # model_cnn = one_hot_cnn(dense_out, max_len)
    # model_hist_cnn = model_cnn.fit(X_train_cnn, Y_train, batch_size=batch_size, epochs=args.epochs,
    #                                verbose=1, validation_data=(X_test_cnn, Y_test))
    # cnn_acc = model_hist_cnn.history['acc'][-1]
    # print("CNN model accuracy: ", cnn_acc)
    #
    # # And make predictions
    # one_hot_cnn_predictions = model_cnn.predict(X_test_cnn)
    #
    # # Using the accuracies create an ensemble
    # accuracies = [lstm_acc, cnn_acc]
    # norm_accuracies = [a / sum(accuracies) for a in accuracies]
    #
    # print("Ensembling with weights: ")
    # for na in norm_accuracies:
    #     print(na)
    # ensembled_predictions = simple_ensembler([lstm_predictions, one_hot_cnn_predictions],
    #                                          norm_accuracies)
    # final_preds = np.argmax(ensembled_predictions, axis=1)
    #
    # # Get the final accuracy
    # print(classification_report(np.argmax(Y_test, axis=1), final_preds))


if __name__ == '__main__':

    # Check file path
    # if args_in.file_path:
    #     validate_existing_filepath(args_in.file_path)

    # if args_in.data_type == 'amazon':
    #     data_in = Amazon_Reviews(args_in.file_path)
    
    args = read_input_args()
    
    with open(os.path.join(args.file_path, 'comments_fasttext_train_2c.txt'), 'r', encoding='utf-8') as train_file, \
        open(os.path.join(args.file_path, 'comments_fasttext_test_2c.txt'), 'r', encoding='utf-8') as test_file:
        _train = [tuple(line.strip().split(' ', 1)) for line in train_file]
        _test = [tuple(line.strip().split(' ', 1)) for line in test_file]
        Y_train, X_train_ = map(list, zip(*_train))
        Y_test, X_test_ = map(list, zip(*_test))
    # ensemble_models(train_label, train_data, test_label, test_data, args_in)

    dense_out = 2
    # split for all models
    # X_train_, X_test_, Y_train, Y_test = train_test_split(data.text, data.labels,
    #                                                      test_size=0.20, random_state=42)

    # Prep data for the LSTM model
    tokenizer = Tokenizer(num_words=args.max_fatures, split=' ')
    tokenizer.fit_on_texts(X_train_)
    X_train = tokenizer.texts_to_sequences(X_train_)
    X_train = pad_sequences(X_train, maxlen=args.max_len)
    X_test = tokenizer.texts_to_sequences(X_test_)
    X_test = pad_sequences(X_test, maxlen=args.max_len)

    # Y_train = to_categorical(Y_train)
    # Y_test = to_categorical(Y_test)
    dic = {
        '__label__1': 0,
        '__label__5': 1
    }
    Y_train = list(map(lambda x: dic[x], Y_train))
    Y_train = to_categorical(Y_train)
    Y_test = list(map(lambda x: dic[x], Y_test))
    Y_test = to_categorical(Y_test)

    # Train the LSTM model
    lstm_model = SentimentLSTM()
    lstm_model.build(args.max_fatures, dense_out, X_train.shape[1], args.embed_dim, args.lstm_out)
    # lstm_model = simple_lstm(max_fatures, dense_out, X_train.shape[1], embed_dim, lstm_out)
    lstm_model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size,
                   verbose=1, validation=(X_test, Y_test))
    lstm_model.save(args.model_path)
    with open(args.model_info_path, 'wb') as fp:
        info = {
            'max_fature': args.max_fatures,
            'max_len':args.max_len,
            'num_of_labels': dense_out,
            'labels_id_to_word': {v: k for k, v in dic.items()},
            'tokenizer':tokenizer,
            'input_length': X_train.shape[1],
            'embed_dim': args.embed_dim,
            'lstm_out': args.lstm_out,
            'dropout': args.dropout,
            'external_embedding_model': args.embedding_model
        }
        pickle.dump(info, fp)
        
    # with open('tokenizer.pkl', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # lstm_predictions = lstm_model.predict(X_test)
