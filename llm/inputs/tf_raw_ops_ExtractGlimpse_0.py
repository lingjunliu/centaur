
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extractglimpse_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 20, 20, 3).astype(np.float32)
    size_tensor = np.array([10, 10]).astype(np.int32)
    offsets_tensor = np.array([[5, 5]]).astype(np.float32)
    input_dict = {"input": input_tensor, "size": size_tensor, "offsets": offsets_tensor, "centered": True, "normalized": True, "uniform_noise": True, "noise": "uniform", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(4, 30, 40, 1).astype(np.float32)
    size_tensor = np.array([15, 20]).astype(np.int32)
    offsets_tensor = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]]).astype(np.float32)
    input_dict = {"input": input_tensor, "size": size_tensor, "offsets": offsets_tensor, "centered": False, "normalized": True, "uniform_noise": False, "noise": "gaussian", "name": "glimpse_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 25, 35, 3).astype(np.float32)
    size_tensor = np.array([8, 12]).astype(np.int32)
    offsets_tensor = np.array([[10, 15], [5, 20]]).astype(np.float32)
    input_dict = {"input": input_tensor, "size": size_tensor, "offsets": offsets_tensor, "centered": True, "normalized": False, "uniform_noise": True, "noise": "uniform", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 10, 15, 1).astype(np.float32)
    size_tensor = np.array([5, 5]).astype(np.int32)
    offsets_tensor = np.array([[0.5, 0.5]]).astype(np.float32)
    input_dict = {"input": input_tensor, "size": size_tensor, "offsets": offsets_tensor, "centered": False, "normalized": True, "uniform_noise": False, "noise": "gaussian", "name": "glimpse_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractGlimpse"] = tf_raw_ops_extractglimpse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractGlimpse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractGlimpse'.")

check_valid('tf.raw_ops.ExtractGlimpse', generated_inputs['tf.raw_ops.ExtractGlimpse'], lib="tf", suffix=0)
