
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tensor_diag_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor (positive integers)
    diagonal = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor (negative integers)
    diagonal = np.array([-1, -2, -3, -4], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor (floats)
    diagonal = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor (complex)
    diagonal = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor (positive integers)
    diagonal = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor (negative integers)
    diagonal = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor (floats)
    diagonal = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D tensor (complex)
    diagonal = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D tensor (int64)
    diagonal = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D tensor (float64)
    diagonal = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "diag_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_linalg_tensor_diag_inputs()
for i in range(len(inputs)):
    inputs[i]['diagonal'] = inputs[i]['diagonal'].numpy()

generated_inputs["tf.linalg.tensor_diag"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.tensor_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tensor_diag'.")

check_valid('tf.linalg.tensor_diag', generated_inputs['tf.linalg.tensor_diag'], lib="tf", suffix=0)
