
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_ignore_errors_inputs():
    list_of_inputs = []

    # Input 1
    log_warning = True
    input_dict = {"log_warning": np.bool_(log_warning).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    log_warning = False
    input_dict = {"log_warning": np.bool_(log_warning).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    log_warning = np.array(True, dtype=np.bool_).item()
    input_dict = {"log_warning": np.array(log_warning, dtype=np.bool_).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    log_warning = np.array(False, dtype=np.bool_).item()
    input_dict = {"log_warning": np.array(log_warning, dtype=np.bool_).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    log_warning = True
    input_dict = {"log_warning": log_warning}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    log_warning = False
    input_dict = {"log_warning": log_warning}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    log_warning = bool(True)
    input_dict = {"log_warning": log_warning}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    log_warning = bool(False)
    input_dict = {"log_warning": log_warning}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    log_warning = np.bool_(True)
    input_dict = {"log_warning": log_warning}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    log_warning = np.bool_(False)
    input_dict = {"log_warning": log_warning}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.ignore_errors"] = tf_data_experimental_ignore_errors_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.ignore_errors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.ignore_errors'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.ignore_errors', generated_inputs['tf.data.experimental.ignore_errors'], lib="tf", suffix=0)
