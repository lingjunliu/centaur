
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def fresnel_cos_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.1, 0.1, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.0, -0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 0.5], [2.0, -1.5]], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, -0.2], [0.3, 0.4]], [[-0.5, 0.6], [0.7, -0.8]]], dtype=np.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-2.5], [3.7]], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5, -1.5]], [[2.0, -2.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.fresnel_cos"] = fresnel_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.special.fresnel_cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.fresnel_cos'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.special.fresnel_cos', generated_inputs['tf.math.special.fresnel_cos'], lib="tf", suffix=0)
