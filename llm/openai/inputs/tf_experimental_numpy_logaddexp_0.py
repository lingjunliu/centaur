
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_logaddexp_inputs():
    list_of_inputs = []

    x1 = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    x2 = np.array([1.5, -2.0, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2 = np.array([[2.0, -4.0, 6.0], [-5.0, 7.0, -9.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.arange(5, dtype=np.float32)
    x2 = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 12.0)
    x2 = np.ones((1, 3, 1), dtype=np.float64) * -2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.full((4, 1), -20.0, dtype=np.float16)
    x2 = np.linspace(-30, 30, 4, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([1e-4, -1e-3, 2.5, -3.5], dtype=np.float16)
    x2 = np.array([-2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([np.inf, -np.inf, 0.0, np.nan], dtype=np.float64)
    x2 = np.array([-np.inf, np.inf, -0.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([-1000.0, -10000.0], dtype=np.float64)
    x2 = np.array([-1000.0, -20000.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    base = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    x1 = base[:, ::2]
    x2 = (-base)[:, ::2]
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([], dtype=np.float32)
    x2 = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([[[0.1, -0.2, 0.3]], [[-0.4, 0.5, -0.6]]], dtype=np.float32)  # (2,1,3)
    x2 = np.array([[[1.0], [-1.0]]], dtype=np.float32)  # (1,2,1)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.random.uniform(-5, 5, size=(2, 3, 1, 4)).astype(np.float32)
    x2 = np.random.uniform(-2, 2, size=(1, 3, 5, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

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
