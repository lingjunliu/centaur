
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_reduce_join_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_)
    reduction_indices = np.array([0], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_)
    reduction_indices = np.array([1], dtype=np.int32)
    keep_dims = True
    separator = "."
    name = "reduce_join_op"
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_)
    reduction_indices = np.array([-1], dtype=np.int32)
    keep_dims = False
    separator = "-"
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_)
    reduction_indices = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_)
    reduction_indices = np.array([1, 0], dtype=np.int32)
    keep_dims = True
    separator = ","
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]], dtype=np.string_)
    reduction_indices = np.array([0], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]], dtype=np.string_)
    reduction_indices = np.array([1, 2], dtype=np.int32)
    keep_dims = True
    separator = "_"
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]], dtype=np.string_)
    reduction_indices = np.array([0, 1, 2], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]], dtype=np.string_)
    reduction_indices = np.array([-1, -2], dtype=np.int32)
    keep_dims = True
    separator = "++"
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]], dtype=np.string_)
    reduction_indices = np.array([2, 0, 1], dtype=np.int32)
    keep_dims = False
    separator = "**"
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: scalar string tensor
    inputs = np.array("test_string", dtype=np.string_)
    reduction_indices = np.array([], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: empty string tensor
    inputs = np.array([["", ""]], dtype=np.string_)
    reduction_indices = np.array([0], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: bytes
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_) # Changed dtype to string as it caused errors
    reduction_indices = np.array([0], dtype=np.int32)
    keep_dims = False
    separator = "" # Changed separator to string as it caused errors
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: bytes with separator
    inputs = np.array([["a", "b"], ["c", "d"]], dtype=np.string_) # Changed dtype to string as it caused errors
    reduction_indices = np.array([1], dtype=np.int32)
    keep_dims = True
    separator = "." # Changed separator to string as it caused errors
    name = "reduce_join_op"
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15: Empty input
    inputs = np.array([], dtype=np.string_)
    reduction_indices = np.array([], dtype=np.int32)
    keep_dims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "reduction_indices": reduction_indices, "keep_dims": keep_dims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReduceJoin"] = tf_raw_ops_reduce_join_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReduceJoin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReduceJoin'.")

check_valid('tf.raw_ops.ReduceJoin', generated_inputs['tf.raw_ops.ReduceJoin'], lib="tf", suffix=0)
