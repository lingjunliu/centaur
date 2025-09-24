
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_max_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D array, reduce along axis 0
    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis1 = np.array([0], dtype=np.int32)
    input_dict1 = {"input": input1, "axis": axis1, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Simple 2D array, reduce along axis 1, keep_dims=True
    input2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis2 = np.array([1], dtype=np.int32)
    input_dict2 = {"input": input2, "axis": axis2, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D array, reduce along multiple axes
    input3 = np.random.rand(2, 3, 4).astype(np.float64)
    axis3 = np.array([0, 2], dtype=np.int32)
    input_dict3 = {"input": input3, "axis": axis3, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D array, reduce along all axes, keep_dims=True
    input4 = np.random.rand(2, 3, 4).astype(np.float64)
    axis4 = np.array([0, 1, 2], dtype=np.int32)
    input_dict4 = {"input": input4, "axis": axis4, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D array, reduce along axis 0
    input5 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis5 = np.array([0], dtype=np.int32)
    input_dict5 = {"input": input5, "axis": axis5, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D array, different data type (int64)
    input6 = np.random.randint(0, 10, size=(2, 3, 4, 5)).astype(np.int64)
    axis6 = np.array([1, 3], dtype=np.int32)
    input_dict6 = {"input": input6, "axis": axis6, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Negative values in input tensor
    input7 = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
    axis7 = np.array([0], dtype=np.int32)
    input_dict7 = {"input": input7, "axis": axis7, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Single element tensor
    input8 = np.array([5], dtype=np.int32)
    axis8 = np.array([0], dtype=np.int32)
    input_dict8 = {"input": input8, "axis": axis8, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Rank 5 tensor
    input9 = np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
    axis9 = np.array([0, 2, 4], dtype=np.int32)
    input_dict9 = {"input": input9, "axis": axis9, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Max"] = tf_raw_ops_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Max'.")

check_valid('tf.raw_ops.Max', generated_inputs['tf.raw_ops.Max'], lib="tf", suffix=0)
