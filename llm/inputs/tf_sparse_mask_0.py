
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_mask_inputs():
    """
    Generates a list of valid inputs for tf.sparse.mask.
    """
    list_of_inputs = []

    # Input 1: Basic case
    values_np_1 = np.arange(40, dtype=np.float32).reshape(4, 10)
    indices_np_1 = np.array([12, 26, 37, 45], dtype=np.int64)
    mask_indices_np_1 = np.array([12, 45], dtype=np.int64)
    dense_shape_np_1 = np.array([100, 10], dtype=np.int64)
    a_1 = tf.IndexedSlices(values=tf.constant(values_np_1), indices=tf.constant(indices_np_1), dense_shape=tf.constant(dense_shape_np_1))
    a_1.size = values_np_1.size
    mask_indices_1 = tf.constant(mask_indices_np_1)
    mask_indices_1.size = mask_indices_np_1.size
    input_dict_1 = {'a': a_1, 'mask_indices': mask_indices_1, 'name': 'basic_mask'}
    list_of_inputs.append(input_dict_1)

    # Input 2: Empty mask
    values_np_2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    indices_np_2 = np.array([5, 10], dtype=np.int32)
    mask_indices_np_2 = np.array([], dtype=np.int32)
    dense_shape_np_2 = np.array([20, 2], dtype=np.int64)
    a_2 = tf.IndexedSlices(values=tf.constant(values_np_2), indices=tf.constant(indices_np_2), dense_shape=tf.constant(dense_shape_np_2))
    a_2.size = values_np_2.size
    mask_indices_2 = tf.constant(mask_indices_np_2)
    mask_indices_2.size = mask_indices_np_2.size
    input_dict_2 = {'a': a_2, 'mask_indices': mask_indices_2, 'name': 'mask_nothing'}
    list_of_inputs.append(input_dict_2)

    # Input 3: Mask everything
    values_np_3 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices_np_3 = np.array([100, 200, 300], dtype=np.int64)
    mask_indices_np_3 = np.array([100, 200, 300], dtype=np.int64)
    dense_shape_np_3 = np.array([500, 2], dtype=np.int64)
    a_3 = tf.IndexedSlices(values=tf.constant(values_np_3), indices=tf.constant(indices_np_3), dense_shape=tf.constant(dense_shape_np_3))
    a_3.size = values_np_3.size
    mask_indices_3 = tf.constant(mask_indices_np_3)
    mask_indices_3.size = mask_indices_np_3.size
    input_dict_3 = {'a': a_3, 'mask_indices': mask_indices_3, 'name': 'mask_everything'}
    list_of_inputs.append(input_dict_3)

    # Input 4: Higher-dimensional values
    values_np_4 = np.random.rand(5, 3, 2).astype(np.float64)
    indices_np_4 = np.array([1, 8, 3, 10, 5], dtype=np.int64)
    mask_indices_np_4 = np.array([8, 10], dtype=np.int64)
    dense_shape_np_4 = np.array([15, 3, 2], dtype=np.int64)
    a_4 = tf.IndexedSlices(values=tf.constant(values_np_4), indices=tf.constant(indices_np_4), dense_shape=tf.constant(dense_shape_np_4))
    a_4.size = values_np_4.size
    mask_indices_4 = tf.constant(mask_indices_np_4)
    mask_indices_4.size = mask_indices_np_4.size
    input_dict_4 = {'a': a_4, 'mask_indices': mask_indices_4, 'name': '3d_values'}
    list_of_inputs.append(input_dict_4)

    # Input 5: Unordered indices
    values_np_5 = np.array([[10], [20], [30], [40]], dtype=np.int32)
    indices_np_5 = np.array([45, 12, 37, 26], dtype=np.int32)
    mask_indices_np_5 = np.array([12, 45], dtype=np.int32)
    dense_shape_np_5 = np.array([50, 1], dtype=np.int64)
    a_5 = tf.IndexedSlices(values=tf.constant(values_np_5), indices=tf.constant(indices_np_5), dense_shape=tf.constant(dense_shape_np_5))
    a_5.size = values_np_5.size
    mask_indices_5 = tf.constant(mask_indices_np_5)
    mask_indices_5.size = mask_indices_np_5.size
    input_dict_5 = {'a': a_5, 'mask_indices': mask_indices_5, 'name': 'unordered_indices'}
    list_of_inputs.append(input_dict_5)

    # Input 6: Single slice which gets masked
    values_np_6 = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    indices_np_6 = np.array([99], dtype=np.int64)
    mask_indices_np_6 = np.array([99], dtype=np.int64)
    dense_shape_np_6 = np.array([100, 5], dtype=np.int64)
    a_6 = tf.IndexedSlices(values=tf.constant(values_np_6), indices=tf.constant(indices_np_6), dense_shape=tf.constant(dense_shape_np_6))
    a_6.size = values_np_6.size
    mask_indices_6 = tf.constant(mask_indices_np_6)
    mask_indices_6.size = mask_indices_np_6.size
    input_dict_6 = {'a': a_6, 'mask_indices': mask_indices_6, 'name': 'single_slice_masked'}
    list_of_inputs.append(input_dict_6)

    # Input 7: Single slice not masked
    values_np_7 = np.array([[1, 2, 3]], dtype=np.int32)
    indices_np_7 = np.array([50], dtype=np.int64)
    mask_indices_np_7 = np.array([10], dtype=np.int64)
    dense_shape_np_7 = np.array([60, 3], dtype=np.int64)
    a_7 = tf.IndexedSlices(values=tf.constant(values_np_7), indices=tf.constant(indices_np_7), dense_shape=tf.constant(dense_shape_np_7))
    a_7.size = values_np_7.size
    mask_indices_7 = tf.constant(mask_indices_np_7)
    mask_indices_7.size = mask_indices_np_7.size
    input_dict_7 = {'a': a_7, 'mask_indices': mask_indices_7, 'name': 'single_slice_not_masked'}
    list_of_inputs.append(input_dict_7)

    # Input 8: Large indices
    values_np_8 = np.array([[1.1], [2.2]], dtype=np.float64)
    indices_np_8 = np.array([1000000, 5000000], dtype=np.int64)
    mask_indices_np_8 = np.array([1000000], dtype=np.int64)
    dense_shape_np_8 = np.array([6000000, 1], dtype=np.int64)
    a_8 = tf.IndexedSlices(values=tf.constant(values_np_8), indices=tf.constant(indices_np_8), dense_shape=tf.constant(dense_shape_np_8))
    a_8.size = values_np_8.size
    mask_indices_8 = tf.constant(mask_indices_np_8)
    mask_indices_8.size = mask_indices_np_8.size
    input_dict_8 = {'a': a_8, 'mask_indices': mask_indices_8, 'name': 'large_indices'}
    list_of_inputs.append(input_dict_8)

    # Input 9: Duplicate indices in `a.indices`
    values_np_9 = np.array([[1], [2], [3], [4]], dtype=np.int32)
    indices_np_9 = np.array([10, 20, 10, 30], dtype=np.int32)
    mask_indices_np_9 = np.array([10], dtype=np.int32)
    dense_shape_np_9 = np.array([40, 1], dtype=np.int64)
    a_9 = tf.IndexedSlices(values=tf.constant(values_np_9), indices=tf.constant(indices_np_9), dense_shape=tf.constant(dense_shape_np_9))
    a_9.size = values_np_9.size
    mask_indices_9 = tf.constant(mask_indices_np_9)
    mask_indices_9.size = mask_indices_np_9.size
    input_dict_9 = {'a': a_9, 'mask_indices': mask_indices_9, 'name': 'duplicate_indices_in_a'}
    list_of_inputs.append(input_dict_9)

    # Input 10: Duplicate values in `mask_indices`
    values_np_10 = np.arange(20, dtype=np.float32).reshape(4, 5)
    indices_np_10 = np.array([1, 5, 9, 13], dtype=np.int64)
    mask_indices_np_10 = np.array([5, 13, 5], dtype=np.int64)
    dense_shape_np_10 = np.array([20, 5], dtype=np.int64)
    a_10 = tf.IndexedSlices(values=tf.constant(values_np_10), indices=tf.constant(indices_np_10), dense_shape=tf.constant(dense_shape_np_10))
    a_10.size = values_np_10.size
    mask_indices_10 = tf.constant(mask_indices_np_10)
    mask_indices_10.size = mask_indices_np_10.size
    input_dict_10 = {'a': a_10, 'mask_indices': mask_indices_10, 'name': 'duplicate_mask_indices'}
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.sparse.mask"] = tf_sparse_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.mask'.")

check_valid('tf.sparse.mask', generated_inputs['tf.sparse.mask'], lib="tf", suffix=0)
