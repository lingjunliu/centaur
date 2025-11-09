
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floor_inputs():
    list_of_inputs = []

    x = np.array([1.3324, -1.5, 5.555, -2.532, 0.99, np.inf], dtype=np.float32)
    name = "basic_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-0.1, 0.0, 0.1], [1.999, 2.001, -2.001]], dtype=np.float64)
    name = "matrix_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3.7, dtype=np.float32)
    name = "scalar32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-3.0001, dtype=np.float64)
    name = "neg_scalar64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, -np.inf, np.inf, -0.0, 0.0], dtype=np.float32)
    name = "nan_inf"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(
        [[[-1.2, -1.8], [1.2, 1.8]], [[0.0, -0.0], [123.5, -123.5]]],
        dtype=np.float16
    )
    name = "tensor3d_fp16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3) / 3.0) - 2.5
    name = "tensor4d_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (np.arange(12, dtype=np.float64).reshape(3, 4) / 2.0) - 3.0
    x = base.T
    name = "non_contiguous64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "empty_vec32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((2, 0, 3), dtype=np.float64)
    name = "empty_tensor_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-12, -1e-12, 1e-8, -1e-8], dtype=np.float64)
    name = "tiny_values64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e10 + 0.9, -1e10 + 0.1, 3.5e5, -3.5e5], dtype=np.float64)
    name = "large_values64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.floor', generated_inputs['tf.math.floor'], lib="tf", suffix=0)
