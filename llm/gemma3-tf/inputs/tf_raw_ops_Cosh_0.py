
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_cosh_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, 0, 1], dtype=np.float64)
    input_dict = {'x': x, 'name': 'cosh_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict = {'x': x, 'name': 'cosh_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.5, 0.0, 2.5], dtype=np.float16)
    input_dict = {'x': x, 'name': 'cosh_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-3.14, 0.0, 3.14], dtype=np.half)
    input_dict = {'x': x, 'name': 'cosh_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.complex128)
    input_dict = {'x': x, 'name': 'cosh_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-100, 0, 100], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0], dtype=np.float64)
    input_dict = {'x': x, 'name': 'cosh_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = tf_raw_ops_cosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cosh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cosh', generated_inputs['tf.raw_ops.Cosh'], lib="tf", suffix=0)
