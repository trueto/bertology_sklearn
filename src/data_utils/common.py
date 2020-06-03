#   Copyright 2020 trueto

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np

def to_numpy(X):
    """
    Convert input to numpy ndarray 将输入转换为numpy的数据格式
    :param X:
    :return:
    """
    if hasattr(X, 'iloc'): # pandas
        return X.values
    elif isinstance(X, list): # list
        return np.array(X)
    elif isinstance(X, np.ndarray): # ndarray
        return X
    else:
        raise ValueError("Unable to handle input type %s " % str(type(X)))

def unpack_text_pairs(X):
    """
    Unpack text pairs
    :param X:
    :return:
    """
    if X.ndim == 1:
        texts_a = X
        texts_b = [None] * len(X)
    else:
        texts_a = X[:, 0]
        texts_b = X[:, 1]

    return texts_a, texts_b

def unpack_data(X, y=None):
    """
    data
    :param X:
    :param y:
    :return:
    """
    X = to_numpy(X)
    texts_a, texts_b = unpack_text_pairs(X)

    if y is not None:
        labels = to_numpy(y)
        return texts_a, texts_b, labels
    else:
        return texts_a, texts_b, None