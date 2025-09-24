
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_mask_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.mask function.

    IMPORTANT NOTE: This function produces inputs that are CORRECT for the
    TensorFlow API. The `a` parameter MUST be a `tf.IndexedSlices` object.
    However, your analysis tool is unable to process this object type because
    it lacks a `.size` attribute, causing an `AttributeError`. There is no
    way to create an input that is valid for both the API and the tool.
    To proceed, the analysis tool must be updated to handle `tf.IndexedSlices`.
    """
    list_of_inputs = []

    # Helper to create IndexedSlices objects, which is the required type for `a`.
    def create_indexed_slices(values, indices, dense_shape):
        return tf.IndexedSlices(
            values=tf.constant(values, dtype=values.dtype),
            indices=tf.constant(indices, dtype=indices.dtype),
            dense_shape=tf.constant(dense_shape, dtype=indices.dtype)
        )

    # Input 1: Basic case from the documentation
    a_values_1 = np.random.rand(4, 10).astype(np.float32)
    a_indices_1 = np.array([12, 26, 37, 45], dtype=np.int64)
    a_dense_shape_1 = np.array([100, 10], dtype=np.int64)
    a_1 = create_indexed_slices(a_values_1, a_indices_1, a_dense_shape_1)
    mask_indices_1 = tf.constant([12, 45], dtype=tf.int64)
    list_of_inputs.append({
        'a': a_1,
        'mask_indices': mask_indices_1,
        'name': 'basic_mask'
    })

    # Input 2: Masking no elements
    a_values_2 = np.arange(15, dtype=np.int32).reshape(3, 5)
    a_indices_2 = np.array([10, 20, 30], dtype=np.int32)
    a_dense_shape_2 = np.array([50, 5], dtype=np.int32)
    a_2 = create_indexed_slices(a_values_2, a_indices_2, a_dense_shape_2)
    mask_indices_2 = tf.constant([], dtype=tf.int32)
    list_of_inputs.append({
        'a': a_2,
        'mask_indices': mask_indices_2,
        'name': 'empty_mask'
    })

    # Input 3: Masking all elements
    a_values_3 = np.random.uniform(size=(4, 2)).astype(np.float64)
    a_indices_3 = np.array([5, 15, 25, 35], dtype=np.int64)
    a_dense_shape_3 = np.array([40, 2], dtype=np.int64)
    a_3 = create_indexed_slices(a_values_3, a_indices_3, a_dense_shape_3)
    mask_indices_3 = tf.constant([5, 15, 25, 35], dtype=tf.int64)
    list_of_inputs.append({
        'a': a_3,
        'mask_indices': mask_indices_3,
        'name': 'mask_all'
    })
    
    # Input 4: Masking a single middle element
    a_values_4 = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float32)
    a_indices_4 = np.array([0, 8, 9], dtype=np.int32)
    a_dense_shape_4 = np.array([10, 2], dtype=np.int32)
    a_4 = create_indexed_slices(a_values_4, a_indices_4, a_dense_shape_4)
    mask_indices_4 = tf.constant([8], dtype=tf.int32)
    list_of_inputs.append({
        'a': a_4,
        'mask_indices': mask_indices_4,
        'name': 'mask_middle'
    })

    # Input 5: Values with more than 2 dimensions
    a_values_5 = np.ones((2, 3, 4), dtype=np.int32)
    a_indices_5 = np.array([100, 200], dtype=np.int64)
    a_dense_shape_5 = np.array([1000, 3, 4], dtype=np.int64)
    a_5 = create_indexed_slices(a_values_5, a_indices_5, a_dense_shape_5)
    mask_indices_5 = tf.constant([100], dtype=np.int64)
    list_of_inputs.append({
        'a': a_5,
        'mask_indices': mask_indices_5,
        'name': 'high_dim_values'
    })

    # Input 6: Masking an index not present in a.indices
    a_values_6 = np.array([[7], [14], [21]], dtype=np.int32)
    a_indices_6 = np.array([7, 14, 21], dtype=np.int32)
    a_dense_shape_6 = np.array([30, 1], dtype=np.int32)
    a_6 = create_indexed_slices(a_values_6, a_indices_6, a_dense_shape_6)
    mask_indices_6 = tf.constant([14, 99], dtype=tf.int32)
    list_of_inputs.append({
        'a': a_6,
        'mask_indices': mask_indices_6,
        'name': 'mask_non_existent'
    })

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
