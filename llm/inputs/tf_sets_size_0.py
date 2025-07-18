
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_size_inputs():
    """
    Generates a list of valid inputs for the tf.sets.size function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D SparseTensor with repeated values, dtype=int32
    values_np_1 = np.array([1, 2, 3, 3, 4], dtype=np.int32)
    a_1 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=np.int64),
        values=values_np_1,
        dense_shape=np.array([2, 4], dtype=np.int64)
    )
    a_1.size = values_np_1.size
    validate_indices_1 = True
    input_dict_1 = {'a': a_1, 'validate_indices': validate_indices_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D SparseTensor, dtype=int64
    values_np_2 = np.array([10, 10, 20, 30, 40, 50, 50], dtype=np.int64)
    a_2 = tf.SparseTensor(
        indices=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.int64),
        values=values_np_2,
        dense_shape=np.array([2, 2, 3], dtype=np.int64)
    )
    a_2.size = values_np_2.size
    validate_indices_2 = True
    input_dict_2 = {'a': a_2, 'validate_indices': validate_indices_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: SparseTensor with an empty set (row 1 is empty)
    values_np_3 = np.array([100, 200], dtype=np.int32)
    a_3 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1]], dtype=np.int64),
        values=values_np_3,
        dense_shape=np.array([2, 5], dtype=np.int64)
    )
    a_3.size = values_np_3.size
    validate_indices_3 = True
    input_dict_3 = {'a': a_3, 'validate_indices': validate_indices_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Completely empty SparseTensor
    values_np_4 = np.array([], dtype=np.int32)
    a_4 = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=values_np_4,
        dense_shape=np.array([3, 3], dtype=np.int64)
    )
    a_4.size = values_np_4.size
    validate_indices_4 = True
    input_dict_4 = {'a': a_4, 'validate_indices': validate_indices_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative integer values, dtype=int16
    values_np_5 = np.array([-1, -2, -1, -2, -1], dtype=np.int16)
    a_5 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int64),
        values=values_np_5,
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    a_5.size = values_np_5.size
    validate_indices_5 = True
    input_dict_5 = {'a': a_5, 'validate_indices': validate_indices_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: validate_indices=False
    values_np_6 = np.array([5, 5, 6, 7, 6], dtype=np.int32)
    a_6 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=np.int64),
        values=values_np_6,
        dense_shape=np.array([2, 5], dtype=np.int64)
    )
    a_6.size = values_np_6.size
    validate_indices_6 = False
    input_dict_6 = {'a': a_6, 'validate_indices': validate_indices_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Higher rank (4D)
    values_np_7 = np.array([1, 2, 3, 3], dtype=np.int32)
    a_7 = tf.SparseTensor(
        indices=np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1]], dtype=np.int64),
        values=values_np_7,
        dense_shape=np.array([2, 1, 2, 3], dtype=np.int64)
    )
    a_7.size = values_np_7.size
    validate_indices_7 = True
    input_dict_7 = {'a': a_7, 'validate_indices': validate_indices_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single set with all unique values
    values_np_8 = np.array([10, 20, 30, 40], dtype=np.int32)
    a_8 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64),
        values=values_np_8,
        dense_shape=np.array([1, 5], dtype=np.int64)
    )
    a_8.size = values_np_8.size
    validate_indices_8 = True
    input_dict_8 = {'a': a_8, 'validate_indices': validate_indices_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: dtype=uint8
    values_np_9 = np.array([10, 20, 10, 30, 40], dtype=np.uint8)
    a_9 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=np.int64),
        values=values_np_9,
        dense_shape=np.array([2, 4], dtype=np.int64)
    )
    a_9.size = values_np_9.size
    validate_indices_9 = True
    input_dict_9 = {'a': a_9, 'validate_indices': validate_indices_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: dtype=int8
    values_np_10 = np.array([-10, 20, -10, -30, 40], dtype=np.int8)
    a_10 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=np.int64),
        values=values_np_10,
        dense_shape=np.array([2, 4], dtype=np.int64)
    )
    a_10.size = values_np_10.size
    validate_indices_10 = True
    input_dict_10 = {'a': a_10, 'validate_indices': validate_indices_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: dtype=uint16
    values_np_11 = np.array([1000, 2000, 1000, 3000], dtype=np.uint16)
    a_11 = tf.SparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        values=values_np_11,
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    a_11.size = values_np_11.size
    validate_indices_11 = True
    input_dict_11 = {'a': a_11, 'validate_indices': validate_indices_11}
    list_of_inputs.append(copy.deepcopy(input_dict_11))


    return list_of_inputs

generated_inputs["tf.sets.size"] = tf_sets_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sets.size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.size'.")

check_valid('tf.sets.size', generated_inputs['tf.sets.size'], lib="tf", suffix=0)
