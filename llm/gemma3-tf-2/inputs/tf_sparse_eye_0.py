
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_eye_inputs():
    list_of_inputs = []

    input_dict = {
        "num_rows": np.int32(5),
        "num_columns": np.int32(5),
        "dtype": np.float32,
        "name": "eye_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(3),
        "num_columns": np.int32(4),
        "dtype": np.float64,
        "name": "eye_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(1),
        "num_columns": np.int32(1),
        "dtype": np.complex64,
        "name": "eye_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(0),
        "num_columns": np.int32(0),
        "dtype": np.float32,
        "name": "eye_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(2),
        "num_columns": np.int32(2),
        "dtype": np.int32,
        "name": "eye_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(6),
        "num_columns": np.int32(3),
        "dtype": np.bool_,
        "name": "eye_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(4),
        "num_columns": np.int32(4),
        "dtype": np.float16,
        "name": "eye_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(7),
        "num_columns": np.int32(7),
        "dtype": np.float32,
        "name": "eye_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(10),
        "num_columns": np.int32(5),
        "dtype": np.float64,
        "name": "eye_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(2),
        "num_columns": np.int32(3),
        "dtype": np.complex128,
        "name": "eye_10"
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
