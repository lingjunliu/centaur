
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_diag_part_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with diagonal elements
    input1 = np.array([[1, 0, 0],
                     [0, 2, 0],
                     [0, 0, 3]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_1",
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with diagonal elements
    input2 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                      [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float64)
    input_dict = {
        "name": "diag_part_input_2",
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with negative values
    input3 = np.array([[-1, 0],
                      [0, -2]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_3",
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with float values
    input4 = np.array([[[[1.5, 2.7], [3.9, 4.1]], [[5.2, 6.8], [7.3, 8.6]]],
                      [[[9.4, 10.1], [11.7, 12.9]], [[13.2, 14.8], [15.3, 16.4]]]], dtype=np.float64)
    input_dict = {
        "name": "diag_part_input_4",
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with complex values
    input5 = np.array([[1+2j, 0],
                      [0, 3+4j]], dtype=np.complex64)
    input_dict = {
        "name": "diag_part_input_5",
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with zero diagonal elements
    input6 = np.array([[0, 0],
                      [0, 0]], dtype=np.int32)
    input_dict = {
        "name": "diag_part_input_6",
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with mixed values
    input7 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                      [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_7",
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with large values
    input8 = np.array([[100, 0],
                      [0, 200]], dtype=np.int64)
    input_dict = {
        "name": "diag_part_input_8",
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with negative diagonal elements
    input9 = np.array([[-1, -2],
                      [-3, -4]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_9",
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with zero diagonal elements
    input10 = np.array([[0, 0],
                       [0, 0]], dtype=np.int32)
    input_dict = {
        "name": "diag_part_input_10",
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DiagPart"] = generate_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)
