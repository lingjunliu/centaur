
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_accumulate_n_inputs():
    list_of_inputs = []

    def create_input_dict(inputs, shape, tensor_dtype, name):
        return {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}

    # Input 1
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    input_dict = create_input_dict(inputs, [2, 2], tf.int32, "accumulate_example_1")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32), np.array([7.0, 8.0, 9.0], dtype=np.float32)]
    input_dict = create_input_dict(inputs, [3], tf.float32, "accumulate_example_2")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [np.array([[-1, -2], [-3, -4]], dtype=np.int64), np.array([[5, 6], [7, 8]], dtype=np.int64)]
    input_dict = create_input_dict(inputs, [2, 2], tf.int64, "accumulate_example_3")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32), np.array([7, 8, 9], dtype=np.int32)]
    input_dict = create_input_dict(inputs, [3], tf.int32, "accumulate_example_4")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64), np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float64)]
    input_dict = create_input_dict(inputs, [2, 2], tf.float64, "accumulate_example_5")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [np.array([1, 2, 3, 4, 5], dtype=np.int32), np.array([6, 7, 8, 9, 10], dtype=np.int32)]
    input_dict = create_input_dict(inputs, [5], tf.int32, "accumulate_example_6")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)]
    input_dict = create_input_dict(inputs, [2, 2], tf.float32, "accumulate_example_7")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64), np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float64)]
    input_dict = create_input_dict(inputs, [2, 2], tf.float64, "accumulate_example_8")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64), np.array([7, 8, 9], dtype=np.int64)]
    input_dict = create_input_dict(inputs, [3], tf.int64, "accumulate_example_9")
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)]
    input_dict = create_input_dict(inputs, [2, 2, 2], tf.int32, "accumulate_example_10")
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.accumulate_n"] = tf_math_accumulate_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)
