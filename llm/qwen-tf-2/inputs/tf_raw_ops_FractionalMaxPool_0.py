
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_fractional_max_pool_inputs():
    list_of_inputs = []
    
    # Input 1
    value = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]], dtype=np.float32)
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "test",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    value = np.array([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]], dtype=np.float32)
    pooling_ratio = [1.0, 1.2, 1.2, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "test2",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    value = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]], dtype=np.float64)
    pooling_ratio = [1.0, 1.7, 1.7, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "test3",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    value = np.array([[[[1., 2., 3., 4.], [5., 6., 7., 8.]]]], dtype=np.int32)
    pooling_ratio = [1.0, 1.4, 1.4, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": True,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "test4",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    value = np.array([[[[1., 2., 3.], [4., 5., 6.]]]], dtype=np.int64)
    pooling_ratio = [1.0, 1.8, 1.8, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": True,
        "seed": 0,
        "seed2": 0,
        "name": "test5",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    value = np.array([[[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]]]], dtype=np.float32)
    pooling_ratio = [1.0, 1.3, 1.3, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": True,
        "deterministic": True,
        "seed": 42,
        "seed2": 43,
        "name": "test6",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    value = np.array([[[[1., 2., 3.], [4., 5., 6.]]]], dtype=np.float64)
    pooling_ratio = [1.0, 1.6, 1.6, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": True,
        "seed": 100,
        "seed2": 200,
        "name": "test7",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    value = np.array([[[[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]]]], dtype=np.int32)
    pooling_ratio = [1.0, 1.9, 1.9, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": False,
        "seed": 50,
        "seed2": 60,
        "name": "test8",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    value = np.array([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]], dtype=np.int64)
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": False,
        "seed": 10,
        "seed2": 20,
        "name": "test9",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    value = np.array([[[[1., 2., 3., 4., 5., 6., 7.], [8., 9., 10., 11., 12., 13., 14.]]]], dtype=np.float32)
    pooling_ratio = [1.0, 1.2, 1.2, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": True,
        "seed": 99,
        "seed2": 100,
        "name": "test10",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.FractionalMaxPool"] = generate_fractional_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FractionalMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalMaxPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FractionalMaxPool', generated_inputs['tf.raw_ops.FractionalMaxPool'], lib="tf", suffix=0)
