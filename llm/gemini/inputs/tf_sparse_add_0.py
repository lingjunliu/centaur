
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_add_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.add function.
    To work around a testing framework that cannot handle tf.SparseTensor objects
    directly, all inputs are provided as dense NumPy arrays. It is assumed the
    framework will convert arrays representing sparse data (i.e., mostly zeros)
    into tf.SparseTensor objects before calling the API.
    """
    list_of_inputs = []

    def to_dense_numpy(indices, values, dense_shape):
        """Helper to create a dense numpy array from sparse components."""
        arr = np.zeros(dense_shape, dtype=values.dtype)
        if not indices.size:
            return arr
        # This handles both 1D and N-D cases
        for idx_tuple, val in zip(indices, values):
            arr[tuple(idx_tuple)] = val
        return arr

    # Input 1: Sparse + Sparse, 2D, float32
    a1 = to_dense_numpy(
        indices=np.array([[0, 1], [2, 3]], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float32),
        dense_shape=(4, 5)
    )
    b1 = to_dense_numpy(
        indices=np.array([[0, 1], [1, 1]], dtype=np.int64),
        values=np.array([3.0, -4.0], dtype=np.float32),
        dense_shape=(4, 5)
    )
    list_of_inputs.append(copy.deepcopy({'a': a1, 'b': b1, 'threshold': np.array(0.0, dtype=np.float32)}))

    # Input 2: Sparse + Dense, 2D, int32
    a2 = to_dense_numpy(
        indices=np.array([[0, 0], [1, 2]], dtype=np.int64),
        values=np.array([5, -10], dtype=np.int32),
        dense_shape=(2, 3)
    )
    b2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({'a': a2, 'b': b2, 'threshold': np.array(0, dtype=np.int32)}))

    # Input 3: Dense + Sparse, 2D, float64
    a3 = np.full((3, 3), 1.5, dtype=np.float64)
    b3 = to_dense_numpy(
        indices=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
        values=np.array([-1.5, 0.0, 3.5], dtype=np.float64),
        dense_shape=(3, 3)
    )
    list_of_inputs.append(copy.deepcopy({'a': a3, 'b': b3, 'threshold': np.array(0.0, dtype=np.float64)}))

    # Input 4: Sparse + Sparse, with positive threshold
    a4 = to_dense_numpy(
        indices=np.array([[0, 1], [2, 3]], dtype=np.int64),
        values=np.array([2.0, -0.2], dtype=np.float32),
        dense_shape=(4, 5)
    )
    b4 = to_dense_numpy(
        indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
        values=np.array([-2.0, 0.1], dtype=np.float32),
        dense_shape=(4, 5)
    )
    list_of_inputs.append(copy.deepcopy({'a': a4, 'b': b4, 'threshold': np.array(0.15, dtype=np.float32)}))

    # Input 5: Sparse + Sparse, 3D
    a5 = to_dense_numpy(
        indices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float64),
        dense_shape=(2, 2, 2)
    )
    b5 = to_dense_numpy(
        indices=np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int64),
        values=np.array([3.0, 4.0], dtype=np.float64),
        dense_shape=(2, 2, 2)
    )
    list_of_inputs.append(copy.deepcopy({'a': a5, 'b': b5, 'threshold': np.array(0.0, dtype=np.float64)}))

    # Input 6: Sparse + Dense, 3D
    a6 = to_dense_numpy(
        indices=np.array([[0, 0, 0]], dtype=np.int64),
        values=np.array([-5], dtype=np.int32),
        dense_shape=(2, 2, 2)
    )
    b6 = np.ones((2, 2, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({'a': a6, 'b': b6, 'threshold': np.array(0, dtype=np.int32)}))

    # Input 7: Sparse + Sparse, 1D
    a7 = to_dense_numpy(
        indices=np.array([[0], [2], [4]], dtype=np.int64),
        values=np.array([10, 20, 30], dtype=np.int32),
        dense_shape=(5,)
    )
    b7 = to_dense_numpy(
        indices=np.array([[1], [2], [3]], dtype=np.int64),
        values=np.array([-5, -20, -15], dtype=np.int32),
        dense_shape=(5,)
    )
    list_of_inputs.append(copy.deepcopy({'a': a7, 'b': b7, 'threshold': np.array(1, dtype=np.int32)}))

    # Input 8: Sparse + Sparse, complex64
    a8 = to_dense_numpy(
        indices=np.array([[0, 0], [1, 1]], dtype=np.int64),
        values=np.array([1+2j, 3+4j], dtype=np.complex64),
        dense_shape=(2, 2)
    )
    b8 = to_dense_numpy(
        indices=np.array([[0, 0], [0, 1]], dtype=np.int64),
        values=np.array([-1-2j, 5+6j], dtype=np.complex64),
        dense_shape=(2, 2)
    )
    list_of_inputs.append(copy.deepcopy({'a': a8, 'b': b8, 'threshold': np.array(0.1, dtype=np.float32)}))
    
    # Input 9: One empty SparseTensor + one non-empty
    a9 = np.zeros((3, 4), dtype=np.float32)
    b9 = to_dense_numpy(
        indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float32),
        dense_shape=(3, 4)
    )
    list_of_inputs.append(copy.deepcopy({'a': a9, 'b': b9, 'threshold': np.array(0.0, dtype=np.float32)}))
    
    # Input 10: Sparse + Dense, complex128
    a10 = to_dense_numpy(
        indices=np.array([[1, 0]], dtype=np.int64),
        values=np.array([10+10j], dtype=np.complex128),
        dense_shape=(2, 1)
    )
    b10 = np.array([[1+1j], [-10-10j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({'a': a10, 'b': b10, 'threshold': np.array(0.001, dtype=np.float64)}))

    return list_of_inputs

generated_inputs["tf.sparse.add"] = tf_sparse_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.add'.")

check_valid('tf.sparse.add', generated_inputs['tf.sparse.add'], lib="tf", suffix=0)
