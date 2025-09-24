
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_diag_part_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict1 = {"input": input1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Larger square matrix
    input2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict2 = {"input": input2, "name": "diag_part_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex64 square matrix
    input3 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict3 = {"input": input3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Higher rank tensor (valid rank 4)
    input4 = np.random.rand(2, 2, 2, 2).astype(np.int64)
    input_dict4 = {"input": input4, "name": "diag_part_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: half data type
    input5 = np.array([[1, 2], [3, 4]], dtype=np.float16)
    input_dict5 = {"input": input5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: float64 data type
    input6 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    input_dict6 = {"input": input6, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: int32 matrix
    input7 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict7 = {"input": input7, "name": "diag_part_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Valid higher dimensional tensor
    input8 = np.random.rand(2,2,2,2).astype(np.int32)
    input_dict8 = {"input": input8, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Complex128 type
    input9 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict9 = {"input": input9, "name": "diag_part_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: int64 type
    input10 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict10 = {"input": input10, "name": "diag_part_op_11"}
    list_of_inputs.append(copy.deepcopy(input_dict10))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DiagPart"] = tf_raw_ops_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)
