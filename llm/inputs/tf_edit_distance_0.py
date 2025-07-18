
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_edit_distance_inputs():
    """
    Generates a list of valid inputs for the tf.edit_distance function.
    tf.edit_distance requires tf.SparseTensor inputs.
    """
    list_of_inputs = []

    # Input 1: Basic case, Rank 2, normalize=True
    hyp1 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        values=np.array([1, 2, 3], dtype=np.int64),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    truth1 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [0, 2], [1, 0]], dtype=np.int64),
        values=np.array([1, 9, 2, 3], dtype=np.int64),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp1,
        'truth': truth1,
        'normalize': True,
        'name': 'basic_rank2_norm'
    }))

    # Input 2: Same as Input 1, but without normalization
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp1,
        'truth': truth1,
        'normalize': False,
        'name': 'basic_rank2_no_norm'
    }))

    # Input 3: Identical sequences
    hyp3 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp3,
        'truth': hyp3,
        'normalize': True,
        'name': 'identical_sequences'
    }))

    # Input 4: Higher rank (Rank 3)
    hyp4 = tf.SparseTensor(
        indices=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=np.int64),
        values=np.array([1, 2, 3], dtype=np.int64),
        dense_shape=np.array([1, 2, 2], dtype=np.int64)
    )
    truth4 = tf.SparseTensor(
        indices=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64),
        values=np.array([1, 9, 3], dtype=np.int64),
        dense_shape=np.array([1, 2, 2], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp4,
        'truth': truth4,
        'normalize': True,
        'name': 'rank3_case'
    }))

    # Input 5: One hypothesis sequence is empty
    hyp5 = tf.SparseTensor(
        indices=np.array([[0, 0]], dtype=np.int64),
        values=np.array([1], dtype=np.int64),
        dense_shape=np.array([2, 1], dtype=np.int64)
    )
    truth5 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 2, 3, 4], dtype=np.int64),
        dense_shape=np.array([2, 2], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp5,
        'truth': truth5,
        'normalize': True,
        'name': 'one_empty_hyp'
    }))

    # Input 6: int32 values
    hyp6 = tf.SparseTensor(
        indices=np.array([[0, 0]], dtype=np.int64),
        values=np.array([100], dtype=np.int32),
        dense_shape=np.array([1, 1], dtype=np.int64)
    )
    truth6 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1]], dtype=np.int64),
        values=np.array([100, 200], dtype=np.int32),
        dense_shape=np.array([1, 2], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp6,
        'truth': truth6,
        'normalize': False,
        'name': 'int32_values'
    }))
    
    # Input 7: Empty truth sequence
    hyp7 = tf.SparseTensor(
        indices=np.array([[0,0]], dtype=np.int64),
        values=np.array([1], dtype=np.int64),
        dense_shape=np.array([1,1], dtype=np.int64)
    )
    truth7 = tf.SparseTensor(
        indices=np.empty((0,2), dtype=np.int64),
        values=np.array([], dtype=np.int64),
        dense_shape=np.array([1,1], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp7,
        'truth': truth7,
        'normalize': True,
        'name': 'empty_truth'
    }))

    # Input 8: Both empty
    hyp8 = tf.SparseTensor(
        indices=np.empty((0,2), dtype=np.int64),
        values=np.array([], dtype=np.int64),
        dense_shape=np.array([1,1], dtype=np.int64)
    )
    truth8 = tf.SparseTensor(
        indices=np.empty((0,2), dtype=np.int64),
        values=np.array([], dtype=np.int64),
        dense_shape=np.array([1,1], dtype=np.int64)
    )
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp8,
        'truth': truth8,
        'normalize': True,
        'name': 'both_empty'
    }))
    
    return list_of_inputs

generated_inputs["tf.edit_distance"] = get_tf_edit_distance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.edit_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.edit_distance'.")

check_valid('tf.edit_distance', generated_inputs['tf.edit_distance'], lib="tf", suffix=0)
