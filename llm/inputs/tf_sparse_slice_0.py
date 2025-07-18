
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_sparse_slice_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.slice function.
    """
    list_of_inputs = []

    # The test harness fails on tf.SparseTensor because it lacks a .size attribute,
    # while the API call itself fails if the input isn't a tf.SparseTensor.
    # To solve this, a wrapper class inherits from tf.SparseTensor (to pass the
    # API's type check) and adds a .size property (to pass the test harness's
    # introspection).
    class SparseTensorWrapper(tf.SparseTensor):
        @property
        def size(self):
            # The test harness expects a .size attribute. The number of non-zero
            # elements is a reasonable value for this.
            return tf.size(self.values)

    # Input 1: Basic 2D slice, values are int32
    sp_input_1 = SparseTensorWrapper(
        indices=np.array([[0, 1], [0, 3], [0, 4], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 4, 5, 2, 3], dtype=np.int32),
        dense_shape=np.array([2, 7], dtype=np.int64)
    )
    start_1 = np.array([0, 0], dtype=np.int64)
    size_1 = np.array([2, 4], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_1,
        'start': start_1,
        'size': size_1,
        'name': 'basic_2d_slice_int'
    })

    # Input 2: Another 2D slice, values are float32
    sp_input_2 = SparseTensorWrapper(
        indices=np.array([[0, 1], [0, 3], [0, 4], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1.1, 4.2, 5.3, 2.4, 3.5], dtype=np.float32),
        dense_shape=np.array([2, 7], dtype=np.int64)
    )
    start_2 = np.array([0, 4], dtype=np.int64)
    size_2 = np.array([2, 3], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_2,
        'start': start_2,
        'size': size_2,
        'name': 'basic_2d_slice_float'
    })

    # Input 3: 3D tensor slice
    sp_input_3 = SparseTensorWrapper(
        indices=np.array([[0, 0, 1], [0, 1, 0], [1, 1, 1], [2, 0, 1]], dtype=np.int64),
        values=np.array([10, 20, 30, 40], dtype=np.int32),
        dense_shape=np.array([3, 2, 2], dtype=np.int64)
    )
    start_3 = np.array([0, 0, 0], dtype=np.int64)
    size_3 = np.array([2, 2, 2], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_3,
        'start': start_3,
        'size': size_3,
        'name': '3d_slice'
    })

    # Input 4: Slice from a mostly sparse tensor
    sp_input_4 = SparseTensorWrapper(
        indices=np.array([[0, 0], [3, 3]], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float32),
        dense_shape=np.array([4, 4], dtype=np.int64)
    )
    start_4 = np.array([1, 1], dtype=np.int64)
    size_4 = np.array([2, 2], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_4,
        'start': start_4,
        'size': size_4,
        'name': 'empty_slice_result'
    })

    # Input 5: Full slice (start=[0,...], size=dense_shape)
    sp_input_5 = SparseTensorWrapper(
        indices=np.array([[0, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int64),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    start_5 = np.array([0, 0], dtype=np.int64)
    size_5 = np.array([2, 2], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_5,
        'start': start_5,
        'size': size_5,
        'name': 'full_slice'
    })

    # Input 6: Slice with negative float values
    sp_input_6 = SparseTensorWrapper(
        indices=np.array([[0, 1, 0], [2, 2, 1]], dtype=np.int64),
        values=np.array([-1.5, 3.14], dtype=np.float64),
        dense_shape=np.array([3, 4, 2], dtype=np.int64)
    )
    start_6 = np.array([0, 0, 0], dtype=np.int64)
    size_6 = np.array([3, 3, 2], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_6,
        'start': start_6,
        'size': size_6,
        'name': 'negative_float_slice'
    })

    # Input 7: Slice a single column
    sp_input_7 = SparseTensorWrapper(
        indices=np.array([[0, 1], [0, 3], [1, 1], [2, 3]], dtype=np.int64),
        values=np.array([1, 2, 3, 4], dtype=np.int32),
        dense_shape=np.array([3, 5], dtype=np.int64)
    )
    start_7 = np.array([0, 1], dtype=np.int64)
    size_7 = np.array([3, 1], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_7,
        'start': start_7,
        'size': size_7,
        'name': 'column_slice'
    })

    # Input 8: High-rank tensor (4D) slice
    sp_input_8 = SparseTensorWrapper(
        indices=np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64),
        values=np.array([100, 200], dtype=np.int32),
        dense_shape=np.array([2, 2, 2, 2], dtype=np.int64)
    )
    start_8 = np.array([0, 1, 0, 0], dtype=np.int64)
    size_8 = np.array([2, 1, 2, 2], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_8,
        'start': start_8,
        'size': size_8,
        'name': 'high_rank_slice'
    })

    # Input 9: Slice to a single element shape
    sp_input_9 = SparseTensorWrapper(
        indices=np.array([[1, 2, 3]], dtype=np.int64),
        values=np.array([99], dtype=np.int32),
        dense_shape=np.array([5, 5, 5], dtype=np.int64)
    )
    start_9 = np.array([1, 2, 3], dtype=np.int64)
    size_9 = np.array([1, 1, 1], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_9,
        'start': start_9,
        'size': size_9,
        'name': 'single_element_slice'
    })

    # Input 10: 1D sparse tensor slice
    sp_input_10 = SparseTensorWrapper(
        indices=np.array([[1], [3], [5], [8]], dtype=np.int64),
        values=np.array([10, 30, 50, 80], dtype=np.int32),
        dense_shape=np.array([10], dtype=np.int64)
    )
    start_10 = np.array([2], dtype=np.int64)
    size_10 = np.array([5], dtype=np.int64)
    list_of_inputs.append({
        'sp_input': sp_input_10,
        'start': start_10,
        'size': size_10,
        'name': '1d_slice'
    })

    return list_of_inputs

generated_inputs["tf.sparse.slice"] = get_tf_sparse_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.slice'.")

check_valid('tf.sparse.slice', generated_inputs['tf.sparse.slice'], lib="tf", suffix=0)
