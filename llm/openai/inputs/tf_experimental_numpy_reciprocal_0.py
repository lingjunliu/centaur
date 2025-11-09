
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_reciprocal_inputs():
    list_of_inputs = []

    # Input 1: scalar float32
    x = np.array(3.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 with negatives and zero
    x = np.array([-1.0, -0.5, 0.0, 2.5], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float16
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32
    x = np.array([[[1.0, 2.0], [-3.0, 4.0]],
                  [[0.5, -0.25], [1e-3, -1e-6]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D complex64 including zero
    x = np.array([1+2j, -3+0j, 0+1j, 0+0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex128
    x = np.array([[1-1j, 2+0j], [0-2j, -0.5+0j]], dtype=np.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 with inf and nan
    x = np.array([np.inf, -np.inf, np.nan, 1.0, -1.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: non-contiguous slice (float64)
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    x = base[:, ::2]
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: empty array (0,) float64
    x = np.array([], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: zero-sized middle dimension, float32
    x = np.empty((2, 0, 3), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: very small magnitudes, float64
    x = np.array([1e-308, -1e-308, 1e-100, -1e-100], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: wide range float32
    x = np.array([1e-2, 1e2, 1e10, -1e-10], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.reciprocal"] = tf_experimental_numpy_reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.reciprocal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.reciprocal', generated_inputs['tf.experimental.numpy.reciprocal'], lib="tf", suffix=0)
