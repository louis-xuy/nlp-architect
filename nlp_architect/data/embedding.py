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
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

from nlp_architect.utils.text import Vocabulary
from nlp_architect.hyperparams import HParams
from nlp_architect.utils import utils


def load_word_embeddings(file_path):
    """
    Loads a word embedding model text file into a word(str) to numpy vector dictionary

    Args:
        file_path (str): path to model file
        emb_size (int): embedding vectors size

    Returns:
        list: a dictionary of numpy.ndarray vectors
        int: detected word embedding vector size
    """
    with open(file_path, encoding='utf-8') as fp:
        word_vectors = {}
        size = None
        for line in fp:
            line_fields = line.split()
            if len(line_fields) < 5:
                continue
            else:
                if line[0] == ' ':
                    word_vectors[' '] = np.asarray(line_fields, dtype='float32')
                else:
                    word_vectors[line_fields[0]] = np.asarray(line_fields[1:], dtype='float32')
                    if size is None:
                        size = len(line_fields[1:])
    return word_vectors, size


def fill_embedding_mat(src_mat, src_lex, emb_lex, emb_size):
    """
    Creates a new matrix from given matrix of int words using the embedding
    model provided.

    Args:
        src_mat (numpy.ndarray): source matrix
        src_lex (dict): source matrix lexicon
        emb_lex (dict): embedding lexicon
        emb_size (int): embedding vector size
    """
    emb_mat = np.zeros((src_mat.shape[0], src_mat.shape[1], emb_size))
    for i, sen in enumerate(src_mat):
        for j, w in enumerate(sen):
            if w > 0:
                w_emb = emb_lex.get(str(src_lex.get(w)).lower())
                if w_emb is not None:
                    emb_mat[i][j] = w_emb
    return emb_mat


def get_embedding_matrix(embeddings: dict, vocab: Vocabulary) -> np.ndarray:
    """
    Generate a matrix of word embeddings given a vocabulary

    Args:
        embeddings (dict): a dictionary of embedding vectors
        vocab (Vocabulary): a Vocabulary

    Returns:
        a 2D numpy matrix of lexicon embeddings
    """
    emb_size = len(next(iter(embeddings.values())))
    mat = np.zeros((vocab.max, emb_size))
    for word, wid in vocab.vocab.items():
        vec = embeddings.get(word.lower(), None)
        if vec is not None:
            mat[wid] = vec
    return mat


class Embedding(object):
    """Embedding class that loads token embedding vectors from file. Token
    embeddings not in the embedding file are initialized as specified in
    :attr:`hparams`.
    Args:
        vocab (dict): A dictionary that maps token strings to integer index.
        read_fn: Callable that takes `(filename, vocab, word_vecs)` and
            returns the updated `word_vecs`. E.g.,
            :func:`~texar.data.embedding.load_word2vec` and
            :func:`~texar.data.embedding.load_glove`.
    """
    def __init__(self, vocab, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

        # Initialize embeddings
        init_fn_kwargs = self._hparams.init_fn.kwargs.todict()
        if "shape" in init_fn_kwargs or "size" in init_fn_kwargs:
            raise ValueError("Argument 'shape' or 'size' must not be "
                             "specified. They are inferred automatically.")
        init_fn = utils.get_function(
            self._hparams.init_fn.type,
            ["numpy.random", "numpy", "texar.custom"])

        try:
            self._word_vecs = init_fn(size=[len(vocab), self._hparams.dim],
                                      **init_fn_kwargs)
        except TypeError:
            self._word_vecs = init_fn(shape=[len(vocab), self._hparams.dim],
                                      **init_fn_kwargs)

        # Optionally read embeddings from file
        if self._hparams.file is not None and self._hparams.file != "":
            read_fn = utils.get_function(
                self._hparams.read_fn,
                ["texar.data.embedding", "texar.data", "texar.custom"])

            self._word_vecs = \
                read_fn(self._hparams.file, vocab, self._word_vecs)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:
        .. role:: python(code)
           :language: python
        .. code-block:: python
            {
                "file": "",
                "dim": 50,
                "read_fn": "load_word2vec",
                "init_fn": {
                    "type": "numpy.random.uniform",
                    "kwargs": {
                        "low": -0.1,
                        "high": 0.1,
                    }
                },
            }
        Here:
        "file" : str
            Path to the embedding file. If not provided, all embeddings are
            initialized with the initialization function.
        "dim": int
            Dimension size of each embedding vector
        "read_fn" : str or callable
            Function to read the embedding file. This can be the function,
            or its string name or full module path. E.g.,
            .. code-block:: python
                "read_fn": texar.data.load_word2vec
                "read_fn": "load_word2vec"
                "read_fn": "texar.data.load_word2vec"
                "read_fn": "my_module.my_read_fn"
            If function string name is used, the function must be in
            one of the modules: :mod:`texar.data` or :mod:`texar.custom`.
            The function must have the same signature as with
            :func:`load_word2vec`.
        "init_fn" : dict
            Hyperparameters of the initialization function used to initialize
            embedding of tokens missing in the embedding
            file.
            The function must accept argument named `size` or `shape` to
            specify the output shape, and return a numpy array of the shape.
            The `dict` has the following fields:
                "type" : str or callable
                    The initialization function. Can be either the function,
                    or its string name or full module path.
                "kwargs" : dict
                    Keyword arguments for calling the function. The function
                    is called with :python:`init_fn(size=[.., ..], **kwargs)`.
        """
        return {
            "file": "",
            "dim": 50,
            "read_fn": "load_word2vec",
            "init_fn": {
                "type": "numpy.random.uniform",
                "kwargs": {
                    "low": -0.1,
                    "high": 0.1,
                },
            },
            "@no_typecheck": ["read_fn", "init_fn"]
        }

    @property
    def word_vecs(self):
        """2D numpy array of shape `[vocab_size, embedding_dim]`.
        """
        return self._word_vecs

    @property
    def vector_size(self):
        """The embedding dimention size.
        """
        return self._hparams.dim