
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_atan2_inputs():
    list_of_inputs = []

    y = np.array([1.0, -1.0], dtype=np.float32)
    x = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "basic_1d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    x = np.array([[1.0, -1.0], [0.0, -2.0]], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "mixed_2d_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([0.5, -0.5, 1.5], dtype=np.float16)
    x = np.array(1.0, dtype=np.float16)
    input_dict = {"y": y, "x": x, "name": "broadcast_scalar_x_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1.0], [-2.0]], dtype=np.float32)
    x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "broadcast_2x1_3_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[[1.0, -1.0], [2.0, -2.0]], [[-3.0, 3.0], [0.0, 0.5]]], dtype=np.float64)
    x = np.array([[[1.0, 1.0], [-2.0, 2.0]], [[3.0, -3.0], [1.0, -0.5]]], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "three_d_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.zeros((4,), dtype=np.float32)
    x = np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "zeros_y_axes_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1e20, -1e20, 3e30], dtype=np.float64)
    x = np.array([1e20, 1e20, -3e30], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "large_vals_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1e-30, -1e-30, 5e-40], dtype=np.float64)
    x = np.array([-1e-30, 1e-30, 5e-40], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "small_vals_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    x = np.array([1.0, -np.inf, np.inf], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "nan_inf_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    input_dict = {"y": y, "x": x, "name": "equal_yx_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = base.T
    x = np.array([[2.0, -2.0, 1.0]], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "transpose_broadcast_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rng = np.random.RandomState(0)
    y = rng.randn(5).astype(np.float32)
    x = rng.randn(5).astype(np.float32)
    input_dict = {"y": y, "x": x, "name": "random_1d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.atan2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.atan2', generated_inputs['tf.math.atan2'], lib="tf", suffix=0)
