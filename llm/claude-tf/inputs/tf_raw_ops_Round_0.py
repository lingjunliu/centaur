
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_round_inputs():
    list_of_inputs = []
    
    x = np.array([1.2, 2.5, 3.7, 4.1], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[-1.5, -2.3], [-3.7, -4.9]], dtype=np.float64)
    input_dict = {"x": x, "name": "round_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.9, 1.1, 1.9], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.5, dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "round_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "name": "round_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    input_dict = {"x": x, "name": "round_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[[1.1, 1.9], [2.1, 2.9]], [[3.1, 3.9], [4.1, 4.9]]]], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100.5, 200.5, 300.5, 400.5], dtype=np.float64)
    input_dict = {"x": x, "name": "round_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, 0.0001, -0.001, -0.0001], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_11"}
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
