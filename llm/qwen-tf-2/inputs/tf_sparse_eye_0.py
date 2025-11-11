
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "num_rows": np.array(3, dtype=np.int32),
        "num_column": np.array(3, dtype=np.int32),
        "dtype": np.float32,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "num_rows": np.array(4, dtype=np.int32),
        "num_column": np.array(5, dtype=np.int32),
        "dtype": np.float64,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "num_rows": np.array(1, dtype=np.int32),
        "num_column": np.array(1, dtype=np.int32),
        "dtype": np.int32,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "num_rows": np.array(0, dtype=np.int32),
        "num_column": np.array(0, dtype=np.int32),
        "dtype": np.float32,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "num_rows": np.array(2, dtype=np.int32),
        "num_column": np.array(2, dtype=np.int32),
        "dtype": np.int64,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "num_rows": np.array(5, dtype=np.int32),
        "num_column": np.array(3, dtype=np.int32),
        "dtype": np.float32,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "num_rows": np.array(10, dtype=np.int32),
        "num_column": np.array(10, dtype=np.int32),
        "dtype": np.float64,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "num_rows": np.array(7, dtype=np.int32),
        "num_column": np.array(7, dtype=np.int32),
        "dtype": np.int32,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "num_rows": np.array(100, dtype=np.int32),
        "num_column": np.array(100, dtype=np.int32),
        "dtype": np.float32,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "num_rows": np.array(20, dtype=np.int32),
        "num_column": np.array(15, dtype=np.int32),
        "dtype": np.float64,
        "name": "eye"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sparse.eye"] = tf_sparse_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.eye'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.eye', generated_inputs['tf.sparse.eye'], lib="tf", suffix=0)
