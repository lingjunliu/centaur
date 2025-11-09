
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softsign_inputs():
    list_of_inputs = []
    rng = np.random.default_rng(42)

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "softsign_case_01"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-10.0, -0.5, 0.0], [0.5, 2.5, 100.0]], dtype=np.float64)
    name = "softsign_case_02"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.linspace(-5, 5, num=24, dtype=np.float16).reshape(2, 3, 4)
    name = "softsign_case_3d_float16"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array(3.14159, dtype=np.float32)
    name = "softsign_case_scalar"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = rng.normal(size=(2, 1, 3, 2)).astype(np.float32)
    name = "softsign_case_4d_random_float32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([], dtype=np.float64)
    name = "softsign_case_empty"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([1e-12, -1e-12, 1e6, -1e6, 1e12, -1e12], dtype=np.float64)
    name = "softsign_case_extremes_float64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0, -1.0], dtype=np.float32)
    name = "softsign_case_inf_nan"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    base = np.arange(-20, 20, dtype=np.float16)
    features = base[::3]
    name = "softsign_case_noncontiguous_slice"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-3.0], [0.0], [2.5], [10.0], [-7.7]], dtype=np.float32)
    name = "softsign_case_column_vector"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-2.0, 0.5, 4.0, -9.0, 1e-3]], dtype=np.float64)
    name = "softsign_case_row_vector_float64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.asfortranarray(np.linspace(-6, 6, 12, dtype=np.float32).reshape(3, 4, order="F"))
    name = "softsign_case_fortran_order"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softsign'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.softsign', generated_inputs['tf.nn.softsign'], lib="tf", suffix=0)
