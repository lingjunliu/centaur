
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_expanddims_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive axis
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with negative axis
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    axis_tensor = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis_tensor = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimension tensor
    input_tensor = np.arange(24).reshape((2, 3, 4)).astype(np.int64)
    axis_tensor = np.array(2, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Expanding at the end
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    axis_tensor = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Expanding with int64 axis
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int64)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Expanding 2D with int64 axis
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis_tensor = np.array(1, dtype=np.int64)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with string type
    input_tensor = np.array(["a", "b", "c"], dtype=np.string_)
    axis_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Empty array
    input_tensor = np.array([], dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Scalar input
    input_tensor = np.array(5, dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Scalar input with negative axis
    input_tensor = np.array(5, dtype=np.int32)
    axis_tensor = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Expand along existing dim
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": axis_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExpandDims"] = tf_raw_ops_expanddims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExpandDims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExpandDims'.")

check_valid('tf.raw_ops.ExpandDims', generated_inputs['tf.raw_ops.ExpandDims'], lib="tf", suffix=0)
