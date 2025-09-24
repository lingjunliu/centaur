
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_diag_part_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    name1 = "diag_part_1"
    k1 = 0
    padding_value1 = np.float32(0.0)
    align1 = "RIGHT_LEFT"
    input_dict1 = {"input": input1, "name": name1, "k": k1, "padding_value": padding_value1, "align": align1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    name2 = "diag_part_2"
    k2 = 1
    padding_value2 = np.int32(0)
    align2 = "LEFT_RIGHT"
    input_dict2 = {"input": input2, "name": name2, "k": k2, "padding_value": padding_value2, "align": align2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    name3 = "diag_part_3"
    k3 = -1
    padding_value3 = np.int64(0)
    align3 = "LEFT_LEFT"
    input_dict3 = {"input": input3, "name": name3, "k": k3, "padding_value": padding_value3, "align": align3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    name4 = "diag_part_4"
    k4 = (0, 1)
    padding_value4 = np.float64(0.0)
    align4 = "RIGHT_RIGHT"
    input_dict4 = {"input": input4, "name": name4, "k": k4, "padding_value": padding_value4, "align": align4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([[1, 2], [3, 4]])
    name5 = "diag_part_5"
    k5 = (-1, 0)
    padding_value5 = np.int8(0)
    align5 = "RIGHT_LEFT"
    input_dict5 = {"input": input5, "name": name5, "k": k5, "padding_value": padding_value5, "align": align5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    name6 = "diag_part_6"
    k6 = 1
    padding_value6 = np.int16(0)
    align6 = "LEFT_RIGHT"
    input_dict6 = {"input": input6, "name": name6, "k": k6, "padding_value": padding_value6, "align": align6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    name7 = "diag_part_7"
    k7 = -1
    padding_value7 = np.int32(1)
    align7 = "LEFT_LEFT"
    input_dict7 = {"input": input7, "name": name7, "k": k7, "padding_value": padding_value7, "align": align7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    name8 = "diag_part_8"
    k8 = (0, 1)
    padding_value8 = np.int64(2)
    align8 = "RIGHT_RIGHT"
    input_dict8 = {"input": input8, "name": name8, "k": k8, "padding_value": padding_value8, "align": align8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
    name9 = "diag_part_9"
    k9 = (-1, 0)
    padding_value9 = np.float32(3.0)
    align9 = "RIGHT_LEFT"
    input_dict9 = {"input": input9, "name": name9, "k": k9, "padding_value": padding_value9, "align": align9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    name10 = "diag_part_10"
    k10 = (-1, 1)
    padding_value10 = np.float64(4.0)
    align10 = "LEFT_RIGHT"
    input_dict10 = {"input": input10, "name": name10, "k": k10, "padding_value": padding_value10, "align": align10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11
    input11 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    name11 = "diag_part_11"
    k11 = (0, 0)
    padding_value11 = np.float32(0.0)
    align11 = "RIGHT_LEFT"
    input_dict11 = {"input": input11, "name": name11, "k": k11, "padding_value": padding_value11, "align": align11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.diag_part"] = tf_linalg_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.diag_part' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.diag_part'.")

check_valid('tf.linalg.diag_part', generated_inputs['tf.linalg.diag_part'], lib="tf", suffix=0)
