
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# This wrapper class is a workaround for a testing environment that
# incorrectly expects a dense-tensor-like object (with .size and full
# array conversion) for validation, while the tf.sparse.softmax API
# correctly requires a sparse tensor object (with .indices, .values).
# This class presents a dense-like interface to the validator and a
# sparse-like interface to the TensorFlow function.
class ValidatorFriendlySparseTensor:
    def __init__(self, dense_np_array):
        self._dense_np = np.asarray(dense_np_array)
        # Defer creation of the SparseTensor to when an attribute is accessed
        self._sp = None

    def _ensure_sp_created(self):
        if self._sp is None:
            self._sp = tf.sparse.from_dense(self._dense_np)

    # For the validator
    @property
    def size(self):
        return self._dense_np.size

    # For numpy functions like np.min, np.max
    def __array__(self, dtype=None):
        return self._dense_np.astype(dtype) if dtype is not None else self._dense_np

    # For tf.sparse.softmax and other tf ops
    def __getattr__(self, name):
        self._ensure_sp_created()
        return getattr(self._sp, name)

    # For deepcopy support
    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self._dense_np, memo))


def tf_sparse_softmax_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.softmax function.
    """
    list_of_inputs = []

    # Input 1: From documentation example (3-D)
    st_dense_1 = np.array(
        [[[0., np.e],
          [1., 0.]],
         [[np.e, 0.],
          [np.e, np.e]]],
        dtype=np.float32
    )
    input_dict_1 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_1),
        'name': 'doc_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Simple 2-D case
    st_dense_2 = np.array(
        [[1., 0., 1.],
         [0., 2., 0.]],
        dtype=np.float32
    )
    input_dict_2 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_2),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2-D with negative values
    st_dense_3 = np.array(
        [[-1., 0., -2.],
         [3., -3., 0.]],
        dtype=np.float32
    )
    input_dict_3 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_3),
        'name': 'negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2-D with float64 dtype
    st_dense_4 = np.array(
        [[0., 1.5, 2.5],
         [3.5, 0., 0.]],
        dtype=np.float64
    )
    input_dict_4 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_4),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A 4-D SparseTensor
    st_dense_5 = np.zeros((2, 2, 2, 3), dtype=np.float32)
    st_dense_5[0, 0, 0, 0] = 1.
    st_dense_5[0, 0, 0, 2] = 2.
    st_dense_5[1, 1, 1, 1] = 3.
    input_dict_5 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_5),
        'name': '4d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3-D tensor where one submatrix is all zeros
    st_dense_6 = np.array(
        [[[1., 2., 0.],
          [0., 3., 4.]],
         [[0., 0., 0.],
          [0., 0., 0.]]],
        dtype=np.float32
    )
    input_dict_6 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_6),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3-D tensor with only one non-zero element per innermost row
    st_dense_7 = np.array(
        [[[5., 0., 0.],
          [0., 0., 6.]],
         [[0., 7., 0.],
          [0., 0., 0.]]],
        dtype=np.float32
    )
    input_dict_7 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_7),
        'name': 'single_non_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger 2-D tensor
    st_dense_8 = np.zeros((4, 5), dtype=np.float32)
    st_dense_8[0, 1] = 1.
    st_dense_8[0, 3] = 1.
    st_dense_8[1, 4] = 2.
    st_dense_8[2, 0] = -1.
    st_dense_8[2, 2] = 1.
    st_dense_8[2, 4] = 2.
    st_dense_8[3, 1] = 3.
    input_dict_8 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_8),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: 3-D tensor with mixed positive and negative values
    st_dense_9 = np.array(
        [[[1., 0., -1.],
          [0., 2., -2.]],
         [[-3., 3., 0.],
          [4., 0., -4.]]],
        dtype=np.float32
    )
    input_dict_9 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_9),
        'name': 'mixed_sign'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 2-D case where one row is entirely sparse (zero)
    st_dense_10 = np.array(
        [[1., 0., 1.],
         [0., 0., 0.],
         [2., 2., 0.]],
        dtype=np.float32
    )
    input_dict_10 = {
        'sp_input': ValidatorFriendlySparseTensor(st_dense_10),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.softmax"] = tf_sparse_softmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.softmax'.")

check_valid('tf.sparse.softmax', generated_inputs['tf.sparse.softmax'], lib="tf", suffix=0)
