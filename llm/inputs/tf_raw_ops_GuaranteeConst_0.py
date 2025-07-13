
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_GuaranteeConst_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "float32_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 tensor with negative values
    input_tensor = np.array([-1, 0, 1, -2, 2], dtype=np.int32)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "int32_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Bool tensor
    input_tensor = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "bool_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 2 float64 tensor
    input_tensor = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "float64_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank 3 int64 tensor
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "int64_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor
    input_tensor = np.array(["hello", "world"])
    input_dict = {"input": tf.convert_to_tensor(input_tensor, dtype=tf.string), "name": "string_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64 tensor
    input_tensor = np.array([1+1j, 2+2j], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "complex64_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 tensor
    input_tensor = np.array([3+3j, 4+4j], dtype=np.complex128)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "complex128_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: UInt8 tensor
    input_tensor = np.array([255, 0, 128], dtype=np.uint8)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "uint8_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 tensor
    input_tensor = np.array([-32768, 0, 32767], dtype=np.int16)
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "name": "int16_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GuaranteeConst"] = tf_raw_ops_GuaranteeConst_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GuaranteeConst' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GuaranteeConst'.")

check_valid('tf.raw_ops.GuaranteeConst', generated_inputs['tf.raw_ops.GuaranteeConst'], lib="tf", suffix=0)
