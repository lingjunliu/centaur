
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_logaddexp_inputs():
    list_of_inputs = []

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([0.0, 1.0, 2.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([-1.0, -2.0, -3.0])
    x2 = np.array([-4.0, -5.0, -6.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[0.0, 1.0], [2.0, 3.0]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    x2 = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.0, -2.0, 3.0])
    x2 = np.array([-4.0, 5.0, -6.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10.0, 20.0, 30.0])
    x2 = np.array([11.0, 21.0, 31.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([-10.0, -20.0, -30.0])
    x2 = np.array([-11.0, -21.0, -31.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.0])
    x2 = np.array([2.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.logaddexp"] = tf_experimental_numpy_logaddexp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.logaddexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logaddexp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.logaddexp', generated_inputs['tf.experimental.numpy.logaddexp'], lib="tf", suffix=0)
