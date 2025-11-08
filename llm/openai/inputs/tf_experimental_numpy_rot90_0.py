
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_rot90_inputs():
    list_of_inputs = []

    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    k = 1
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.array([[1.1, -2.2, 3.3], [4.4, 5.5, -6.6]], dtype=np.float64)
    k = -1
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    k = 2
    axes = (1, 2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    k = 3
    axes = (0, 2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 2 * 3 * 4, dtype=np.uint8).reshape(2, 2, 3, 4)
    k = 1
    axes = (2, 3)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.array([[1 + 2j, 3 - 4j], [5 + 0j, -6 + 1j]], dtype=np.complex64)
    k = 2
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = (np.arange(6) % 2 == 0).reshape(3, 2)
    k = 3
    axes = (-2, -1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(4, dtype=np.int64).reshape(4, 1)
    k = 7
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 1 * 3 * 4 * 5, dtype=np.float16).reshape(2, 1, 3, 4, 5)
    k = -2
    axes = (0, 3)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(20, dtype=np.int8).reshape(4, 5)
    k = 1
    axes = (1, 0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.empty((0, 3), dtype=np.float32)
    k = 1
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    base = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    m = base + 1j * base
    k = -3
    axes = (-3, -1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.rot90"] = tf_experimental_numpy_rot90_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.rot90'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.rot90', generated_inputs['tf.experimental.numpy.rot90'], lib="tf", suffix=0)
