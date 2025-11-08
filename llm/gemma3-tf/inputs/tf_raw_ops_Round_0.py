
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_round_inputs():
    list_of_inputs = []
    
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {'x': x, 'name': 'round_test_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.5, -2.5, -3.5, -4.5], dtype=np.float64)
    input_dict = {'x': x, 'name': 'round_test_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.2, 2.7], [3.1, 4.9]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'round_test_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64)
    input_dict = {'x': x, 'name': 'round_test_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.int32)
    input_dict = {'x': x, 'name': 'round_test_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    input_dict = {'x': x, 'name': 'round_test_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.half)
    input_dict = {'x': x, 'name': 'round_test_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Round"] = tf_raw_ops_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Round'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Round', generated_inputs['tf.raw_ops.Round'], lib="tf", suffix=0)
