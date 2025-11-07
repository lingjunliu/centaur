
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def matrix_transpose_inputs():
    list_of_inputs = []
    
    # Input 1: 2D real tensor
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: 2D complex tensor with conjugate=True
    a = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: 3D tensor (batched matrix)
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: 2D tensor with negative values
    a = np.array([[-1, -2, -3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: 2D tensor with zero values
    a = np.array([[0, 2, 3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: 3D tensor with different shapes
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: 2D tensor with float values
    a = np.array([[1.5, 2.7], [3.1, 4.8]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: 3D tensor with mixed types
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: 2D tensor with complex values
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: 2D tensor with mixed values (float and int)
    a = np.array([[1.0, 2], [3.0, 4]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.linalg.matrix_transpose"] = matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.matrix_transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.matrix_transpose', generated_inputs['tf.linalg.matrix_transpose'], lib="tf", suffix=0)
