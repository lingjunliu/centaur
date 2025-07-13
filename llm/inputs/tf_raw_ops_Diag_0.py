
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    diagonal = np.array([1, 2, 3, 4], dtype=np.int32)
    name = "diag_1"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float data type
    diagonal = np.array([1.0, 2.5, 3.7], dtype=np.float32)
    name = "diag_2"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int64 data type
    diagonal = np.array([1000000000, 2000000000], dtype=np.int64)
    name = "diag_3"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty array
    diagonal = np.array([], dtype=np.float32)
    name = "diag_4"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64
    diagonal = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    name = "diag_5"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Bfloat16 (needs conversion for numpy)
    diagonal = np.array([1, 2], dtype=np.float16)
    name = "diag_6"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Half (float16)
    diagonal = np.array([3, 4], dtype=np.float16)
    name = "diag_7"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: negative integers
    diagonal = np.array([-1, -2, -3], dtype=np.int32)
    name = "diag_8"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 0 tensor (scalar)
    diagonal = np.array(5, dtype=np.int32)
    name = "diag_9"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    diagonal = np.array([1 + 1j, 2 - 2j], dtype=np.complex128)
    name = "diag_10"
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Diag"] = tf_raw_ops_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Diag'.")

check_valid('tf.raw_ops.Diag', generated_inputs['tf.raw_ops.Diag'], lib="tf", suffix=0)
