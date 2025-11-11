
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_lrn_inputs():
    list_of_inputs = []
    
    # Input 1: 4D tensor with float32 dtype
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with half dtype
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float16)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 3,
        "bias": 2.0,
        "alpha": 0.5,
        "beta": 0.75,
        "name": "lrn_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with bfloat16 dtype
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_tensor = input_tensor.astype(np.float16)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 7,
        "bias": 0.5,
        "alpha": 2.0,
        "beta": 0.25,
        "name": "lrn_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with negative values
    input_tensor = np.random.rand(1, 2, 3, 4).astype(np.float32)
    input_tensor[0, 0, 0, 0] = -1.0
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with zero values
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_tensor[0, 0, 0, 0] = 0.0
    input_dict = {
        "input": input_tensor,
        "depth_radius": 3,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with different shape
    input_tensor = np.random.rand(1, 2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 2,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 1,
        "bias": 0.1,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 4,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.LRN"] = generate_lrn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LRN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LRN'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LRN', generated_inputs['tf.raw_ops.LRN'], lib="tf", suffix=0)
