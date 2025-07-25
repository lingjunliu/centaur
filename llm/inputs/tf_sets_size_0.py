
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

    # Input 1: Basic 2D case with integer values
    a_indices_1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=np.int64)
    a_values_1 = np.array([1, 2, 3, 4, 3], dtype=np.int32)
    a_dense_shape_1 = np.array([2, 5], dtype=np.int64)
    a_1 = tf.SparseTensor(indices=a_indices_1, values=a_values_1, dense_shape=a_dense_shape_1)
    input_dict_1 = {
        "a": a_1,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D case with repeated values to test uniqueness
    a_indices_2 = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int64)
    a_values_2 = np.array([10, 20, 10, 30, 30], dtype=np.int32)
    a_dense_shape_2 = np.array([2, 3], dtype=np.int64)
    a_2 = tf.SparseTensor(indices=a_indices_2, values=a_values_2, dense_shape=a_dense_shape_2)
    input_dict_2 = {
        "a": a_2,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D case
    a_indices_3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]], dtype=np.int64)
    a_values_3 = np.array([1, 2, 3, 4, 1], dtype=np.int32)
    a_dense_shape_3 = np.array([2, 2, 3], dtype=np.int64)
    a_3 = tf.SparseTensor(indices=a_indices_3, values=a_values_3, dense_shape=a_dense_shape_3)
    input_dict_3 = {
        "a": a_3,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Case with empty sets
    a_indices_4 = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
    a_values_4 = np.array([1, 2, 3], dtype=np.int32)
    a_dense_shape_4 = np.array([3, 2], dtype=np.int64)
    a_4 = tf.SparseTensor(indices=a_indices_4, values=a_values_4, dense_shape=a_dense_shape_4)
    input_dict_4 = {
        "a": a_4,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Completely empty SparseTensor
    a_indices_5 = np.empty((0, 2), dtype=np.int64)
    a_values_5 = np.array([], dtype=np.int32)
    a_dense_shape_5 = np.array([3, 4], dtype=np.int64)
    a_5 = tf.SparseTensor(indices=a_indices_5, values=a_values_5, dense_shape=a_dense_shape_5)
    input_dict_5 = {
        "a": a_5,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: uint8 values
    a_indices_6 = np.array([[0, 0], [0, 1], [0, 2], [1, 0]], dtype=np.int64)
    a_values_6 = np.array([1, 2, 1, 3], dtype=np.uint8)
    a_dense_shape_6 = np.array([2, 4], dtype=np.int64)
    a_6 = tf.SparseTensor(indices=a_indices_6, values=a_values_6, dense_shape=a_dense_shape_6)
    input_dict_6 = {
        "a": a_6,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: String values are not supported by the test harness, skipping.
    # New valid input: int16
    a_indices_7 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    a_values_7 = np.array([100, -200, 100, 300], dtype=np.int16)
    a_dense_shape_7 = np.array([2, 3], dtype=np.int64)
    a_7 = tf.SparseTensor(indices=a_indices_7, values=a_values_7, dense_shape=a_dense_shape_7)
    input_dict_7 = {
        "a": a_7,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    

    # Input 8: validate_indices=False
    input_dict_8 = {
        "a": a_1,
        "validate_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large dense_shape relative to indices
    a_indices_9 = np.array([[0, 0]], dtype=np.int64)
    a_values_9 = np.array([100], dtype=np.int32)
    a_dense_shape_9 = np.array([5, 5], dtype=np.int64)
    a_9 = tf.SparseTensor(indices=a_indices_9, values=a_values_9, dense_shape=a_dense_shape_9)
    input_dict_9 = {
        "a": a_9,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensor
    a_indices_10 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int64)
    a_values_10 = np.array([1, 1, 2, 3], dtype=np.int32)
    a_dense_shape_10 = np.array([2, 2, 2, 2], dtype=np.int64)
    a_10 = tf.SparseTensor(indices=a_indices_10, values=a_values_10, dense_shape=a_dense_shape_10)
    input_dict_10 = {
        "a": a_10,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: int64 values
    a_indices_11 = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    a_values_11 = np.array([9223372036854775807, -1, 9223372036854775807], dtype=np.int64)
    a_dense_shape_11 = np.array([2, 2], dtype=np.int64)
    a_11 = tf.SparseTensor(indices=a_indices_11, values=a_values_11, dense_shape=a_dense_shape_11)
    input_dict_11 = {
        "a": a_11,
        "validate_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Monkey-patch the size attribute for the test harness
    for input_dict in list_of_inputs:
        st = input_dict['a']
        if isinstance(st, tf.SparseTensor):
            st.size = st.values.numpy().size

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
