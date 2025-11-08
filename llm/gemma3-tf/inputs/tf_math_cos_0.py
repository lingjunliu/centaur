
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_cos_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
    input_dict = {"x": x, "name": "cos_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    input_dict = {"x": x, "name": "cos_special"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x, "name": "cos_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0])
    input_dict = {"x": x, "name": "cos_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x, "name": "cos_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x, "name": "cos_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j])
    input_dict = {"x": x, "name": "cos_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {"x": x, "name": "cos_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "name": "cos_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 1.0, 2.0], dtype=np.complex64)
    input_dict = {"x": x, "name": "cos_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.cos"] = tf_math_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cos'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.cos', generated_inputs['tf.math.cos'], lib="tf", suffix=0)
