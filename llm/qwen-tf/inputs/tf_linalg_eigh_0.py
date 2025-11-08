
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with positive values
    tensor1 = np.array([[3., 1., 2.], [1., 4., 5.], [2., 5., 6.]])
    input_dict1 = {
        'tensor': tensor1,
        'name': 'test1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor with negative values
    tensor2 = np.array([[[1., -2.], [-2., 1.]], [[3., 4.], [4., 3.]]])
    input_dict2 = {
        'tensor': tensor2,
        'name': 'test2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor with zero values
    tensor3 = np.array([[[[1., 0.], [0., 1.]], [[2., 0.], [0., 2.]]], [[[3., 0.], [0., 3.]], [[4., 0.], [0., 4.]]]])
    input_dict3 = {
        'tensor': tensor3,
        'name': 'test3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D tensor with float values
    tensor4 = np.array([[1.5, 2.7], [2.7, 1.5]])
    input_dict4 = {
        'tensor': tensor4,
        'name': 'test4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor with mixed values
    tensor5 = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]])
    input_dict5 = {
        'tensor': tensor5,
        'name': 'test5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor with complex values
    tensor6 = np.array([[1.+2j, 3.+4j], [3.+4j, 1.+2j]])
    input_dict6 = {
        'tensor': tensor6,
        'name': 'test6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D tensor with very small values
    tensor7 = np.array([[0.001, 0.002], [0.002, 0.001]])
    input_dict7 = {
        'tensor': tensor7,
        'name': 'test7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 2D tensor with large values
    tensor8 = np.array([[1000., 2000.], [2000., 1000.]])
    input_dict8 = {
        'tensor': tensor8,
        'name': 'test8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 2D tensor with identity matrix
    tensor9 = np.array([[1., 0.], [0., 1.]])
    input_dict9 = {
        'tensor': tensor9,
        'name': 'test9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 3D tensor with symmetric matrix
    tensor10 = np.array([[[[1., 2.], [2., 1.]], [[3., 4.], [4., 3.]]], [[[5., 6.], [6., 5.]], [[7., 8.], [8., 7.]]]])
    input_dict10 = {
        'tensor': tensor10,
        'name': 'test10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.linalg.eigh"] = generate_inputs

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
