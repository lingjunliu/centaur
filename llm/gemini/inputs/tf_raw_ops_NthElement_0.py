
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NthElement_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'reverse': False,
        'name': "nth_element_1",
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'n': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'reverse': True,
        'name': "nth_element_2",
        'input': np.array([[5, 2, 9, 1], [3, 8, 4, 7]], dtype=np.int32),
        'n': np.array(2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'reverse': False,
        'name': "nth_element_3",
        'input': np.array([[[1.5, -2.3], [0.0, 4.1]], [[-1.1, 2.2], [3.3, -4.4]]], dtype=np.float64),
        'n': np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'reverse': False,
        'name': "nth_element_4",
        'input': np.array([10, 20, 30, 40, 50], dtype=np.int64),
        'n': np.array(4, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'reverse': True,
        'name': "nth_element_5",
        'input': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'n': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'reverse': False,
        'name': "nth_element_6",
        'input': np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        'n': np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'reverse': True,
        'name': "nth_element_7",
        'input': np.array([[-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.int64),
        'n': np.array(3, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'reverse': False,
        'name': "nth_element_8",
        'input': np.array([3.14, 2.71, 1.41], dtype=np.float64),
        'n': np.array(2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'reverse': True,
        'name': "nth_element_9",
        'input': np.array([[[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]], dtype=np.float32),
        'n': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'reverse': False,
        'name': "nth_element_10",
        'input': np.array([1000, 2000, 3000, 4000], dtype=np.int32),
        'n': np.array(2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NthElement"] = tf_raw_ops_NthElement_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.NthElement' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NthElement'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.NthElement', generated_inputs['tf.raw_ops.NthElement'], lib="tf", suffix=0)
