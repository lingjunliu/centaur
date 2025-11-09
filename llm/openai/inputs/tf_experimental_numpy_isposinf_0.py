
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_experimental_numpy_isposinf_inputs():
    list_of_inputs = []

    x = np.array(np.inf, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-np.inf, 0.0, np.inf, np.nan, 1.5, -2.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[np.inf, -np.inf, 5.0], [3.4e38, np.inf, -np.inf]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.random.randn(2, 3, 4).astype(np.float64)
    x[0, 0, 0] = np.inf
    x[1, 2, 3] = np.inf
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([65504.0, np.inf, -np.inf, 1.0, -0.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[np.inf]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.zeros((1, 2, 1, 3), dtype=np.float64)
    x[0, 0, 0, 1] = np.inf
    x[0, 1, 0, 2] = np.inf
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([0.0, -0.0, 1e-308, -1e-308, np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([np.float32(3.402823e38), np.float32(np.inf), np.float32(-np.inf), np.float32(0.0)], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-1.0, -2.0, -np.inf, -1e10], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.linspace(-10, 10, 11).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = tf_experimental_numpy_isposinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isposinf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isposinf', generated_inputs['tf.experimental.numpy.isposinf'], lib="tf", suffix=0)
