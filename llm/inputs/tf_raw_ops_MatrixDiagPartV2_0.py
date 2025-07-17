
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_diag_part_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic matrix with k=0
    input1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k1 = np.array(0).astype(np.int32)
    padding_value1 = np.array(0).astype(np.int32)
    input_dict1 = {"input": input1, "k": k1, "padding_value": padding_value1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: k=1 (superdiagonal)
    input2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k2 = np.array(1).astype(np.int32)
    padding_value2 = np.array(0).astype(np.int32)
    input_dict2 = {"input": input2, "k": k2, "padding_value": padding_value2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: k=-1 (subdiagonal)
    input3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k3 = np.array(-1).astype(np.int32)
    padding_value3 = np.array(0).astype(np.int32)
    input_dict3 = {"input": input3, "k": k3, "padding_value": padding_value3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: k=(0,1) (band)
    input4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k4 = np.array([0, 1]).astype(np.int32)
    padding_value4 = np.array(0).astype(np.int32)
    input_dict4 = {"input": input4, "k": k4, "padding_value": padding_value4, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: k=(-1,0) (band)
    input5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k5 = np.array([-1, 0]).astype(np.int32)
    padding_value5 = np.array(0).astype(np.int32)
    input_dict5 = {"input": input5, "k": k5, "padding_value": padding_value5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different padding value
    input6 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k6 = np.array(0).astype(np.int32)
    padding_value6 = np.array(-1).astype(np.int32)
    input_dict6 = {"input": input6, "k": k6, "padding_value": padding_value6, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger k values
    input7 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    k7 = np.array(2).astype(np.int32)
    padding_value7 = np.array(0).astype(np.int32)
    input_dict7 = {"input": input7, "k": k7, "padding_value": padding_value7, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D tensor
    input8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    k8 = np.array(0).astype(np.int32)
    padding_value8 = np.array(0).astype(np.int32)
    input_dict8 = {"input": input8, "k": k8, "padding_value": padding_value8, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Float input
    input9 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).astype(np.float32)
    k9 = np.array(0).astype(np.int32)
    padding_value9 = np.array(0.0).astype(np.float32)
    input_dict9 = {"input": input9, "k": k9, "padding_value": padding_value9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Complex input
    input10 = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]]).astype(np.complex64)
    k10 = np.array(0).astype(np.int32)
    padding_value10 = np.array(0+0j).astype(np.complex64)
    input_dict10 = {"input": input10, "k": k10, "padding_value": padding_value10, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiagPartV2"] = tf_raw_ops_matrix_diag_part_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiagPartV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagPartV2'.")

check_valid('tf.raw_ops.MatrixDiagPartV2', generated_inputs['tf.raw_ops.MatrixDiagPartV2'], lib="tf", suffix=0)
