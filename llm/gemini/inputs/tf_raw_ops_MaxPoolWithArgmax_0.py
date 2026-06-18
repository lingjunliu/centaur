
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_MaxPoolWithArgmax_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': False,
        'name': "maxpool_1",
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': True,
        'name': "maxpool_2",
        'input': np.random.randint(-10, 10, size=(2, 3, 3, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': True,
        'name': "maxpool_3",
        'input': np.random.randn(1, 5, 5, 3).astype(np.float64),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': False,
        'name': "maxpool_4",
        'input': np.random.randint(-50, 50, size=(3, 8, 8, 4)).astype(np.int32),
        'ksize': [1, 4, 4, 1],
        'strides': [1, 4, 4, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': False,
        'name': "maxpool_5",
        'input': np.random.randint(0, 255, size=(1, 2, 2, 1)).astype(np.int64),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': True,
        'name': "maxpool_6",
        'input': np.random.randn(2, 6, 6, 2).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': False,
        'name': "maxpool_7",
        'input': np.random.randint(-5, 5, size=(1, 4, 4, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': True,
        'name': "maxpool_8",
        'input': np.random.randint(0, 1000, size=(4, 4, 4, 1)).astype(np.int64),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': False,
        'name': "maxpool_9",
        'input': np.random.randn(1, 10, 10, 3).astype(np.float32),
        'ksize': [1, 5, 5, 1],
        'strides': [1, 3, 3, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': True,
        'name': "maxpool_10",
        'input': np.random.randn(2, 2, 2, 2).astype(np.float64),
        'ksize': [1, 1, 1, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolWithArgmax"] = tf_raw_ops_MaxPoolWithArgmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolWithArgmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolWithArgmax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPoolWithArgmax', generated_inputs['tf.raw_ops.MaxPoolWithArgmax'], lib="tf", suffix=0)
