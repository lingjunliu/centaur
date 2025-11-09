
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_floor_divide_inputs():
    list_of_inputs = []

    x1 = np.array([1, 2, 3, -4, 5], dtype=np.int32)
    x2 = np.array([2, -3, 4, 5, -6], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[10, -20, 30], [40, -50, 60]], dtype=np.int64)
    x2 = np.array([[3, 4, -5], [-6, 7, 8]], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1.5], [-2.5], [3.75]], dtype=np.float32)
    x2 = np.array([[2.0, -3.0, 4.0, 5.0]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array(123, dtype=np.int32)
    x2 = np.array(7, dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[100, 200], [250, 255]], dtype=np.uint8)
    x2 = np.array([[3, 5], [7, 9]], dtype=np.uint8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([-10.0, -20.5, 30.2, 40.8], dtype=np.float64)
    x2 = np.array(-2.5, dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[[12, 15, -18], [20, -25, 30]], [[-35, 40, 45], [50, 55, -60]]], dtype=np.int16)
    x2 = np.array([[[3, -4, 5], [6, 7, -8]]], dtype=np.int16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([7, -8, 9], dtype=np.int32)
    x2 = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[10], [20], [-30], [40]], dtype=np.int8)
    x2 = np.array([3, -4, 5, 6], dtype=np.int8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, -6.5], [7.25, -8.75, 9.125]], dtype=np.float16)
    x2 = np.array([[-2.0, 3.0, -4.0], [5.0, -6.0, 7.0], [-8.0, 9.0, -10.0]], dtype=np.float16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([-100, -1, 0, 1, 100], dtype=np.int64)
    x2 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array(10.5, dtype=np.float32)
    x2 = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_experimental_numpy_floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.floor_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.floor_divide'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.floor_divide', generated_inputs['tf.experimental.numpy.floor_divide'], lib="tf", suffix=0)
