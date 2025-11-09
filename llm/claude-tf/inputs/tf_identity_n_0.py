
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []
    
    input_dict = {"input": np.array([[1.0, 2.0], [3.0, 4.0]]), "name": "identity_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([1, 2, 3, 4, 5]), "name": "identity_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([[-1.5, -2.5], [-3.5, -4.5]]), "name": "identity_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "name": "identity_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([1, -2, 3, -4, 5]), "name": "identity_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array(42.0), "name": "identity_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([[0, 0], [0, 0], [0, 0]]), "name": "identity_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.ones((2, 3, 4)), "name": "identity_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([100.5, 200.5, 300.5]), "name": "identity_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([[[[1, 2], [3, 4]]]]), "name": "identity_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
