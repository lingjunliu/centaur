
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NthElement_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    input_val = np.array([[3, 1, 4, 1, 5], [1, 6, 1, 8, 9]], dtype=np.int32)
    n_val = np.array(2, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "basic_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floats with reverse
    input_val = np.array([[3.0, 1.0, 4.0, 1.0, 5.0], [1.0, 6.0, 1.0, 8.0, 9.0]], dtype=np.float32)
    n_val = np.array(1, dtype=np.int32)
    reverse_val = True
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "float_reverse"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single vector
    input_val = np.array([5, 2, 8, 1, 9], dtype=np.int32)
    n_val = np.array(3, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "single_vector"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (int64)
    input_val = np.array([[3, 1, 4, 1, 5], [1, 6, 1, 8, 9]], dtype=np.int64)
    n_val = np.array(2, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (float64) and reverse
    input_val = np.array([[3.0, 1.0, 4.0, 1.0, 5.0], [1.0, 6.0, 1.0, 8.0, 9.0]], dtype=np.float64)
    n_val = np.array(1, dtype=np.int32)
    reverse_val = True
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "float64_reverse"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape
    input_val = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    n_val = np.array(1, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "diff_shape"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: n = 0
    input_val = np.array([[3, 1, 4, 1, 5], [1, 6, 1, 8, 9]], dtype=np.int32)
    n_val = np.array(0, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "n_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: n close to max
    input_val = np.array([[3, 1, 4, 1, 5], [1, 6, 1, 8, 9]], dtype=np.int32)
    n_val = np.array(4, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "n_high"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16
    input_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    n_val = np.array(1, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8
    input_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    n_val = np.array(1, dtype=np.int32)
    reverse_val = False
    input_dict = {"input": input_val, "n": n_val, "reverse": reverse_val, "name": "uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NthElement"] = tf_raw_ops_NthElement_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NthElement' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NthElement'.")

check_valid('tf.raw_ops.NthElement', generated_inputs['tf.raw_ops.NthElement'], lib="tf", suffix=0)
