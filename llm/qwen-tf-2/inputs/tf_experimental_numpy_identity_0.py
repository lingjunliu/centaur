
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_identity_inputs():
    list_of_inputs = []
    
    # Input 1, valid - integer n=3 with float dtype
    input_dict = {
        "n": 3,
        "dtype": np.float64
    }
    list_of_inputs.append(input_dict)
    
    # Input 2, valid - integer n=5 with int dtype
    input_dict = {
        "n": 5,
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    # Input 3, valid - integer n=1 with bool dtype
    input_dict = {
        "n": 1,
        "dtype": np.bool_
    }
    list_of_inputs.append(input_dict)
    
    # Input 4, valid - integer n=0 with float dtype (edge case)
    input_dict = {
        "n": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid - integer n=2 with float dtype
    input_dict = {
        "n": 2,
        "dtype": np.float32
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid - integer n=4 with int dtype
    input_dict = {
        "n": 4,
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 7, valid - integer n=10 with float dtype
    input_dict = {
        "n": 10,
        "dtype": np.float64
    }
    list_of_inputs.append(input_dict)
    
    # Input 8, valid - integer n=7 with bool dtype
    input_dict = {
        "n": 7,
        "dtype": np.bool_
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, valid - integer n=100 with float dtype
    input_dict = {
        "n": 100,
        "dtype": np.float64
    }
    list_of_inputs.append(input_dict)
    
    # Input 10, valid - integer n=50 with int dtype
    input_dict = {
        "n": 50,
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.identity"] = tf_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.identity', generated_inputs['tf.experimental.numpy.identity'], lib="tf", suffix=0)
