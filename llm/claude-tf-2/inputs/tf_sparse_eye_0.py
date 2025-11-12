
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []
    
    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "dtype": np.float32,
        "name": "eye1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 10,
        "num_columns": 5,
        "dtype": np.float32,
        "name": "eye2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 3,
        "num_columns": 8,
        "dtype": np.float64,
        "name": "eye3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 1,
        "num_columns": 1,
        "dtype": np.float32,
        "name": "eye4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 100,
        "num_columns": 100,
        "dtype": np.float32,
        "name": "eye5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 7,
        "num_columns": 7,
        "dtype": np.int32,
        "name": "eye6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 4,
        "num_columns": 6,
        "dtype": np.int64,
        "name": "eye7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 2,
        "num_columns": 20,
        "dtype": np.float32,
        "name": "eye8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 15,
        "num_columns": 15,
        "dtype": np.float16,
        "name": "eye9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 25,
        "num_columns": 10,
        "dtype": np.float64,
        "name": "eye10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 50,
        "num_columns": 3,
        "dtype": np.float32,
        "name": "eye11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 10,
        "num_columns": 1,
        "dtype": np.float32,
        "name": "eye12"
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
