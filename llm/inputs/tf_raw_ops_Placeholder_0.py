
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_placeholder_inputs():
    list_of_inputs = []

    # Input 1: Simple int32 scalar
    input_dict = {
        "dtype": tf.int32,
        "shape": [],
        "name": "int32_scalar"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 vector
    input_dict = {
        "dtype": tf.float32,
        "shape": [5],
        "name": "float32_vector"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String matrix
    input_dict = {
        "dtype": tf.string,
        "shape": [2, 3],
        "name": "string_matrix"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor with unknown shape
    input_dict = {
        "dtype": tf.bool,
        "shape": None,
        "name": "bool_unknown_shape"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64 tensor with 3 dimensions
    input_dict = {
        "dtype": tf.int64,
        "shape": [1, 4, 2],
        "name": "int64_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Complex64 vector
    input_dict = {
        "dtype": tf.complex64,
        "shape": [10],
        "name": "complex64_vector"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float16 matrix
    input_dict = {
        "dtype": tf.float16,
        "shape": [4, 4],
        "name": "float16_matrix"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: UInt8 tensor
    input_dict = {
        "dtype": tf.uint8,
        "shape": [2, 2, 2, 2],
        "name": "uint8_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: BFloat16 scalar
    input_dict = {
        "dtype": tf.bfloat16,
        "shape": [],
        "name": "bfloat16_scalar"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Quint8 Tensor
    input_dict = {
        "dtype": tf.quint8,
        "shape": [3, 4],
        "name": "quint8_matrix"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Placeholder"] = tf_raw_ops_placeholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Placeholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Placeholder'.")

check_valid('tf.raw_ops.Placeholder', generated_inputs['tf.raw_ops.Placeholder'], lib="tf", suffix=0)
