
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_diag_inputs():
    list_of_inputs = []

    # Input 1: Rank 1, int32
    diagonal = np.array([1, 2, 3, 4], dtype=np.int32)
    name = "diag_int32"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rank 1, float32
    diagonal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    name = "diag_float32"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rank 1, complex64
    diagonal = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64)
    name = "diag_complex64"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 1, int64
    diagonal = np.array([1, 2, 3, 4], dtype=np.int64)
    name = "diag_int64"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank 1, float64
    diagonal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "diag_float64"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank 1, bfloat16
    diagonal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).astype(np.float32) #bfloat16 not directly supported in numpy, emulate with float16 conversion
    name = "diag_bfloat16"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rank 1, half
    diagonal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    name = "diag_half"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank 0, int32 (scalar)
    diagonal = np.array(5, dtype=np.int32)
    name = "diag_scalar_int32"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 0, float32 (scalar)
    diagonal = np.array(5.5, dtype=np.float32)
    name = "diag_scalar_float32"
    input_dict = {"diagonal": diagonal, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 1, negative values, int32
    diagonal = np.array([-1, -2, -3, -4], dtype=np.int32)
    name = "diag_negative_int32"
    input_dict = {"diagonal": diagonal, "name": name}
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
