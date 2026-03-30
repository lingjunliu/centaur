
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_enumerate_dataset_inputs():
    list_of_inputs = []

    # Input 1: start = 0
    start = np.int64(0)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: start = 1
    start = np.int64(1)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: start = 10
    start = np.int64(10)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: start = -1
    start = np.int64(-1)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: start = large positive number
    start = np.int64(2**31 - 1)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: start = large negative number
    start = np.int64(-(2**31))
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: start as a numpy array (scalar)
    start = np.array(5, dtype=np.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: start as numpy array
    start = np.array([2], dtype=np.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: start = small positive number
    start = np.int64(5)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: start = small negative number
    start = np.int64(-5)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for i, input_dict in enumerate(list_of_inputs):
        try:
            output = run_api(api, input_dict, cpu=True, lib=lib)
        except Exception as e:
            print(f"Exception {type(e)}:{e} at input {i} suffix {suffix}")
            raise
    return True

def run_api(api, input_dict, cpu=True, lib="tf"):
    if lib == "torch":
        if "dtype" in input_dict:
            input_dict["dtype"] = getattr(torch, input_dict["dtype"])
        if "layout" in input_dict:
            input_dict["layout"] = getattr(torch, input_dict["layout"])

    api_func = eval(api)
    return api_func(**input_dict)

generated_inputs["tf.data.experimental.enumerate_dataset"] = tf_data_experimental_enumerate_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.enumerate_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.enumerate_dataset'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.enumerate_dataset', generated_inputs['tf.data.experimental.enumerate_dataset'], lib="tf", suffix=0)
