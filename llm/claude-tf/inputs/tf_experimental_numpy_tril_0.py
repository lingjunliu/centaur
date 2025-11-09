
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_tril_inputs():
    list_of_inputs = []
    
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2], [3, 4], [5, 6]])
    k = 5
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    k = -3
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[5]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([1, 2, 3, 4, 5])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1.5, 2.3, 3.7], [4.1, 5.9, 6.2], [7.8, 8.4, 9.6]])
    k = 2
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tf_experimental_numpy_tril_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.tril'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.tril', generated_inputs['tf.experimental.numpy.tril'], lib="tf", suffix=0)
