
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_diag_part_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[1, 2], [3, 4]])
    input_dict = {"input": input1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"input": input2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input3 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    input_dict = {"input": input3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input4 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    input_dict = {"input": input4, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input5 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    input_dict = {"input": input5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input6 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
    input_dict = {"input": input6, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"input": input7, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input8 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {"input": input8, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input9 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"input": input9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input10 = np.array([[[1, 0, 0, 0],
                     [0, 2, 0, 0],
                     [0, 0, 3, 0],
                     [0, 0, 0, 4]],
                    [[5, 0, 0, 0],
                     [0, 6, 0, 0],
                     [0, 0, 7, 0],
                     [0, 0, 0, 8]]])
    input_dict = {"input": input10, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiagPart"] = tf_raw_ops_matrix_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagPart'.")

check_valid('tf.raw_ops.MatrixDiagPart', generated_inputs['tf.raw_ops.MatrixDiagPart'], lib="tf", suffix=0)
