
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_Counter_inputs():
    list_of_inputs = []

    input_dict = {
        "start": np.int64(0),
        "step": np.int64(1),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(5),
        "step": np.int64(2),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(-3),
        "step": np.int64(1),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(10),
        "step": np.int64(-2),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(100),
        "step": np.int64(5),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(-10),
        "step": np.int64(-1),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(0),
        "step": np.int64(-1),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(7),
        "step": np.int64(3),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(-5),
        "step": np.int64(2),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(15),
        "step": np.int64(-3),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_Counter_inputs()

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
