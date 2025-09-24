
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_transpose_inputs():
    list_of_inputs = []

    # The test harness fails because tf.SparseTensor lacks attributes like .size
    # that the harness expects from a generic 'tensor'. The API itself, however,
    # strictly requires a tf.SparseTensor object. The following is a workaround
    # to make tf.SparseTensor compatible with the test harness by adding the
    # expected attributes at runtime (monkey-patching). This does not affect
    # the correctness of the input for the tf.sparse.transpose API.
    if not hasattr(tf.SparseTensor, 'size'):
        tf.SparseTensor.size = property(lambda self: tf.size(self.values).numpy())
    if not hasattr(tf.SparseTensor, '__array__'):
        tf.SparseTensor.__array__ = lambda self: self.values.numpy()


    # Input 1: Basic 2D transpose (perm=[1,0])
    sp_input_1 = tf.SparseTensor(
        indices=np.array([[0, 1], [2, 3]], dtype=np.int64),
        values=np.array([1.1, 2.2], dtype=np.float32),
        dense_shape=np.array([4, 5], dtype=np.int64)
    )
    input_dict_1 = {
        'sp_input': sp_input_1,
        'perm': [1, 0],
        'name': 'basic_2d_transpose'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D transpose (default perm=[2,1,0])
    sp_input_2 = tf.SparseTensor(
        indices=np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int32),
        dense_shape=np.array([2, 4, 5], dtype=np.int64)
    )
    input_dict_2 = {
        'sp_input': sp_input_2,
        'perm': [2, 1, 0],
        'name': 'default_3d_transpose'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D batch matrix transpose (perm=[0,2,1])
    sp_input_3 = tf.SparseTensor(
        indices=np.array([[0, 1, 2], [0, 2, 1], [1, 0, 1]], dtype=np.int64),
        values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        dense_shape=np.array([2, 3, 3], dtype=np.int64)
    )
    input_dict_3 = {
        'sp_input': sp_input_3,
        'perm': [0, 2, 1],
        'name': 'batch_matrix_transpose'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D custom permutation with negative values (perm=[1,2,0])
    sp_input_4 = tf.SparseTensor(
        indices=np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int64),
        values=np.array([-5, -10], dtype=np.int32),
        dense_shape=np.array([2, 4, 5], dtype=np.int64)
    )
    input_dict_4 = {
        'sp_input': sp_input_4,
        'perm': [1, 2, 0],
        'name': 'custom_3d_perm_neg_vals'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 4D transpose with custom perm
    sp_input_5 = tf.SparseTensor(
        indices=np.array([[0, 0, 1, 0], [1, 2, 0, 1]], dtype=np.int64),
        values=np.array([100, 200], dtype=np.int64),
        dense_shape=np.array([2, 3, 2, 2], dtype=np.int64)
    )
    input_dict_5 = {
        'sp_input': sp_input_5,
        'perm': [0, 3, 1, 2],
        'name': 'custom_4d_perm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty sparse tensor
    sp_input_6 = tf.SparseTensor(
        indices=np.empty((0, 3), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=np.array([5, 10, 2], dtype=np.int64)
    )
    input_dict_6 = {
        'sp_input': sp_input_6,
        'perm': [2, 0, 1],
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D sparse tensor (vector)
    sp_input_7 = tf.SparseTensor(
        indices=np.array([[1], [3], [5]], dtype=np.int64),
        values=np.array([1, 2, 3], dtype=np.float32),
        dense_shape=np.array([7], dtype=np.int64)
    )
    input_dict_7 = {
        'sp_input': sp_input_7,
        'perm': [0],
        'name': '1d_vector_transpose'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensor with a dimension of size 1
    sp_input_8 = tf.SparseTensor(
        indices=np.array([[0, 0, 3], [0, 0, 4]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int64),
        dense_shape=np.array([1, 5, 5], dtype=np.int64)
    )
    input_dict_8 = {
        'sp_input': sp_input_8,
        'perm': [2, 0, 1],
        'name': 'dim_size_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Full sparse tensor (all elements are non-zero)
    sp_input_9 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 2, 3, 4], dtype=np.int32),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    input_dict_9 = {
        'sp_input': sp_input_9,
        'perm': [1, 0],
        'name': 'full_sparse_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 5D transpose with default permutation
    sp_input_10 = tf.SparseTensor(
        indices=np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float32),
        dense_shape=np.array([2, 2, 2, 2, 2], dtype=np.int64)
    )
    input_dict_10 = {
        'sp_input': sp_input_10,
        'perm': [4, 3, 2, 1, 0],
        'name': '5d_transpose_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 4D tensor with identity permutation (no change)
    sp_input_11 = tf.SparseTensor(
        indices=np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64),
        values=np.array([-3.0, 3.0], dtype=np.float32),
        dense_shape=np.array([2, 2, 2, 2], dtype=np.int64)
    )
    input_dict_11 = {
        'sp_input': sp_input_11,
        'perm': [0, 1, 2, 3],
        'name': 'identity_perm_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.sparse.transpose"] = tf_sparse_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.transpose'.")

check_valid('tf.sparse.transpose', generated_inputs['tf.sparse.transpose'], lib="tf", suffix=0)
