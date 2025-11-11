
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []
    
    v = np.array([1, 2, 3, 4])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([1.5, 2.5, 3.5])
    k = 2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([10, 20, 30])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([-5, -10, -15, -20])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([42])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    k = 3
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
    k = -2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    v = np.array([0, 0, 0])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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
