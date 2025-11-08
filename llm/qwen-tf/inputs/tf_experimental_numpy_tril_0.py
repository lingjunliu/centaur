
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tril_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with k=0
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with k=1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with k=-1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with k=0
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with k=1
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with k=-1
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensor with k=0
    m = np.array([1, 2, 3])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with k=1
    m = np.array([1, 2, 3])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D tensor with k=-1
    m = np.array([1, 2, 3])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tf_tril_inputs()

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
