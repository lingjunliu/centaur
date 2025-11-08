
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_selu_inputs():
    list_of_inputs = []

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "selu_vec_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-1.5, 0.0, 2.5], [3.3, -4.4, 5.5]], dtype=np.float64)
    name = "selu_matrix_f64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[[0.1, -0.1], [0.5, -0.5]], [[-1.2, 1.2], [2.0, -2.0]]], dtype=np.float16)
    name = "selu_3d_f16"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array(0.0, dtype=np.float32)
    name = "selu_scalar_zero"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4, 1)
    name = "selu_4d_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([], dtype=np.float32)
    name = "selu_empty_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    base = np.linspace(-3.0, 3.0, 12, dtype=np.float32)
    features = base[::2]
    name = "selu_noncontig_vec"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-10.0, -1.0, 0.0], [1.0, 5.5, 10.0]], dtype=np.float32)
    name = "selu_2x3_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([-100.0, -20.0, 20.0, 100.0], dtype=np.float32)
    name = "selu_extreme_vals"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    name = "selu_nan_inf"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    arr = np.arange(16, dtype=np.float64) - 8.0
    features = (arr / 4.0).reshape(1, 2, 2, 2, 2)
    name = "selu_5d_f64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([1e-4, -1e-4, 1e-2, -1e-2], dtype=np.float16)
    name = "selu_small_f16"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    return list_of_inputs

generated_inputs["tf.nn.selu"] = tf_nn_selu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.selu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.selu', generated_inputs['tf.nn.selu'], lib="tf", suffix=0)
