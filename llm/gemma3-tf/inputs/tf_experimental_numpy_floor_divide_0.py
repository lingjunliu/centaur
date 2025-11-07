
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_floor_divide_inputs():
    list_of_inputs = []

    x1 = np.array([10, 20, 30])
    x2 = np.array([3, 4, 5])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([-10, -20, -30])
    x2 = np.array([3, 4, 5])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([-3, -4, -5])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[-10, -20], [-30, -40]])
    x2 = np.array([3, 4])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([2])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.5, 2.5, 3.5])
    x2 = np.array([1, 2, 3])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([1, 1, 1])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([2, 2, 2])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([3, 4, 6])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[2, 3], [4, 5]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10])
    x2 = np.array([3])
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_floor_divide_inputs()

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
