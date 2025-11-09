
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_log1p_inputs():
    list_of_inputs = []

    x = np.array([0.0, 0.5, 1.0, 5.0], dtype=np.float32)
    name = "log1p_float32_1d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[-0.99999994, -0.5], [0.0, 10.0]], dtype=np.float64)
    name = "log1p_float64_2d_neg"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = (np.arange(-6, 6, dtype=np.float16) / 10).reshape(2, 3, 2)
    name = "log1p_float16_3d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1+2j, -0.5+0.3j, -2+5j], dtype=np.complex64)
    name = "log1p_complex64_1d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0+0j, -1+1j], [3-4j, -0.999+0.001j]], dtype=np.complex128)
    name = "log1p_complex128_2d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(-0.9, dtype=np.float32)
    name = "log1p_scalar_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-10, 1e10, -0.9999999999], dtype=np.float64)
    name = "log1p_large_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "log1p_empty"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    base = np.arange(-1.0, 5.0, 0.5, dtype=np.float32)
    x = base[::2]
    name = "log1p_noncontig_slice"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0], dtype=np.float32)
    name = "log1p_nan_inf"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((2, 1, 3, 4), dtype=np.float32)
    x[0, 0, 1, 2] = -0.5
    x[1, 0, 2, 3] = 10.0
    name = "log1p_4d_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-1.0+1e-12j, -1.0-1e-12j, 0.0+0.0j], dtype=np.complex128)
    name = "log1p_complex_branch"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log1p'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.log1p', generated_inputs['tf.math.log1p'], lib="tf", suffix=0)
