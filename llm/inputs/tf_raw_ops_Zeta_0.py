
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_zeta_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([2.0], dtype=np.float32)
    q = np.array([1.0], dtype=np.float32)
    name = "zeta_basic_float32"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64
    x = np.array([2.0], dtype=np.float64)
    q = np.array([1.0], dtype=np.float64)
    name = "zeta_basic_float64"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-element float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    name = "zeta_multi_float32"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-element float64
    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    name = "zeta_multi_float64"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: q < 0
    x = np.array([2.0], dtype=np.float32)
    q = np.array([-0.5], dtype=np.float32)
    name = "zeta_negative_float32"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: q < 0
    x = np.array([2.0], dtype=np.float64)
    q = np.array([-0.5], dtype=np.float64)
    name = "zeta_negative_float64"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: q close to 0
    x = np.array([2.0], dtype=np.float32)
    q = np.array([0.001], dtype=np.float32)
    name = "zeta_zero_float32"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: q close to 0
    x = np.array([2.0], dtype=np.float64)
    q = np.array([0.001], dtype=np.float64)
    name = "zeta_zero_float64"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array float32
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    q = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    name = "zeta_2d_float32"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array float64
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    q = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
    name = "zeta_2d_float64"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: large x and q
    x = np.array([100.0], dtype=np.float32)
    q = np.array([50.0], dtype=np.float32)
    name = "zeta_large_float32"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: large x and q
    x = np.array([100.0], dtype=np.float64)
    q = np.array([50.0], dtype=np.float64)
    name = "zeta_large_float64"
    input_dict = {"x": x, "q": q, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Zeta"] = tf_raw_ops_zeta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Zeta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Zeta'.")

check_valid('tf.raw_ops.Zeta', generated_inputs['tf.raw_ops.Zeta'], lib="tf", suffix=0)
