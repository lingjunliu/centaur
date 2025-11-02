
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isfinite_inputs():
    list_of_inputs = []

    x = np.array([1.0, -2.5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[np.inf, -np.inf], [np.nan, 3.14]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([np.nan, np.inf, -1.23], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[[1.0, -0.0], [np.inf, -np.inf]], [[np.nan, 2.0], [3.5, -4.5]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    arr = np.linspace(-10, 10, 20, dtype=np.float64).reshape(4, 5)
    arr[2, 4] = -np.inf
    x = arr[:, ::2]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.asfortranarray(np.array([[1.0, np.nan], [np.inf, -3.0]], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.full((2, 3, 4, 5), fill_value=np.nan, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([1e308, -1e308, 1e-308, np.inf, -np.inf, np.nan], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    finfo16 = np.finfo(np.float16)
    x = np.array([finfo16.max, finfo16.tiny, -finfo16.max, np.nan, np.inf, -np.inf], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_experimental_numpy_isfinite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isfinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isfinite'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isfinite', generated_inputs['tf.experimental.numpy.isfinite'], lib="tf", suffix=0)
