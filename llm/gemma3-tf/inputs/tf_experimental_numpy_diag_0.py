
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []
    
    v1 = np.array([[1, 2], [3, 4]])
    k1 = 0
    input_dict1 = {"v": v1, "k": k1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    v2 = np.array([1, 2, 3])
    k2 = 0
    input_dict2 = {"v": v2, "k": k2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    v3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k3 = 1
    input_dict3 = {"v": v3, "k": k3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    v4 = np.array([[1, 2], [3, 4]])
    k4 = -1
    input_dict4 = {"v": v4, "k": k4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    v5 = np.array([1, 2])
    k5 = 0
    input_dict5 = {"v": v5, "k": k5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    v6 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    k6 = 2
    input_dict6 = {"v": v6, "k": k6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    v7 = np.array([[1, 2, 3], [4, 5, 6]])
    k7 = -1
    input_dict7 = {"v": v7, "k": k7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    v8 = np.array([1, 2, 3, 4, 5])
    k8 = 1
    input_dict8 = {"v": v8, "k": k8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    v9 = np.array([[1, 2], [3, 4]])
    k9 = 0
    input_dict9 = {"v": v9, "k": k9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    v10 = np.array([[1, 2, 3], [4, 5, 6]])
    k10 = -2
    input_dict10 = {"v": v10, "k": k10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.diag', generated_inputs['tf.experimental.numpy.diag'], lib="tf", suffix=0)
