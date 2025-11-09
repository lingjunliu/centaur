
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eigh_inputs():
    list_of_inputs = []
    
    tensor = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "eigh_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[1.0, 0.5], [0.5, 2.0]], [[3.0, 1.0], [1.0, 4.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.eye(3, dtype=np.float64)
    input_dict = {"tensor": tensor, "name": "identity_eigh"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1.0, 2.0], [2.0, -1.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[5.0, 1.0, 0.0, 0.5], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 3.0, 1.0], [0.5, 0.0, 1.0, 2.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "large_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.random.randn(2, 2, 3, 3).astype(np.float32)
    tensor = (tensor + np.transpose(tensor, (0, 1, 3, 2))) / 2
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
    input_dict = {"tensor": tensor, "name": "diagonal_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[0.01, 0.005], [0.005, 0.02]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[5.0]], [[10.0]], [[15.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "batch_1x1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.eigh"] = tf_linalg_eigh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.eigh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.eigh', generated_inputs['tf.linalg.eigh'], lib="tf", suffix=0)
