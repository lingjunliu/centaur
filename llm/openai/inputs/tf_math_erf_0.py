
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_erf_inputs():
    list_of_inputs = []

    x = np.array(0.0, dtype=np.float32)
    name = "erf_scalar_zero_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(-1.5, dtype=np.float64)
    name = "erf_scalar_neg_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1.0, 2.0, 3.0, -0.5], dtype=np.float32)
    name = "erf_1d_mixed_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]], dtype=np.float16)
    name = "erf_2d_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.uniform(-3, 3, size=(2, 2, 3)).astype(np.float32)
    name = "erf_3d_random_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "erf_empty_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-10.0, -5.0, -2.0, 2.0, 5.0, 10.0], dtype=np.float64)
    name = "erf_large_magnitudes_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    name = "erf_inf_nan_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(0, 10, dtype=np.float32)[::2]
    name = "erf_noncontiguous_stride_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float16).reshape(-1, 1)
    name = "erf_column_vector_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 2, 2, 3)
    name = "erf_4d_linspace_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-12, -1e-12], dtype=np.float64)
    name = "erf_tiny_values_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.erf', generated_inputs['tf.math.erf'], lib="tf", suffix=0)
