
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_counter_inputs():
    list_of_inputs = []
    
    # Input 1: Default values
    input_dict = {
        "start": 0,
        "step": 1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Custom start value
    input_dict = {
        "start": 10,
        "step": 1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Custom step value
    input_dict = {
        "start": 0,
        "step": 5,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative step
    input_dict = {
        "start": 10,
        "step": -1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: int32 dtype
    input_dict = {
        "start": 0,
        "step": 1,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative start
    input_dict = {
        "start": -5,
        "step": 1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large step
    input_dict = {
        "start": 0,
        "step": 100,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Negative start and step
    input_dict = {
        "start": -10,
        "step": -2,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large start value
    input_dict = {
        "start": 1000,
        "step": 10,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Custom start, step, and int32 dtype
    input_dict = {
        "start": 2,
        "step": 5,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Zero start with negative step
    input_dict = {
        "start": 0,
        "step": -10,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Large negative start
    input_dict = {
        "start": -1000,
        "step": 50,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_counter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.Counter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.Counter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.Counter', generated_inputs['tf.data.experimental.Counter'], lib="tf", suffix=0)
