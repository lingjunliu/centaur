
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_pad_inputs():
    list_of_inputs = []

    # Input 1: Basic constant padding
    tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    paddings = np.array([[1, 1], [2, 2]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(0, dtype=np.int32)
    name = "basic_constant"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Reflect padding
    tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    paddings = np.array([[1, 1], [1, 1]], dtype=np.int32)
    mode = "REFLECT"
    constant_values = np.array(0, dtype=np.int32)
    name = "reflect_padding"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Symmetric padding
    tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    paddings = np.array([[1, 1], [1, 1]], dtype=np.int32)
    mode = "SYMMETRIC"
    constant_values = np.array(0, dtype=np.int32)
    name = "symmetric_padding"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Constant padding with non-zero constant values
    tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    paddings = np.array([[1, 0], [0, 1]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(-1, dtype=np.int32)
    name = "constant_nonzero"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor constant padding
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    paddings = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(0, dtype=np.int32)
    name = "3d_tensor"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tensor
    tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    paddings = np.array([[2, 1]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(0, dtype=np.int32)
    name = "1d_tensor"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float tensor with float constant values
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    paddings = np.array([[1, 1], [1, 1]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(2.5, dtype=np.float32)
    name = "float_tensor"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Paddings with zero values
    tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    paddings = np.array([[0, 0], [0, 0]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(0, dtype=np.int32)
    name = "zero_padding"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger padding values, fixing dimension issue
    tensor = np.array([[1, 2]], dtype=np.int32)
    paddings = np.array([[1, 1], [0, 0]], dtype=np.int32)
    mode = "CONSTANT"
    constant_values = np.array(7, dtype=np.int32)
    name = "larger_padding"
    input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Padding string tensor. Removed due to dtype errors
    #tensor = np.array([["a", "b"], ["c", "d"]], dtype=np.str_)
    #paddings = np.array([[1, 1], [1, 1]], dtype=np.int32)
    #mode = "CONSTANT"
    #constant_values = np.array("pad", dtype=np.str_)
    #name = "string_padding"
    #input_dict = {"tensor": tensor, "paddings": paddings, "mode": mode, "constant_values": constant_values, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.pad"] = tf_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.pad'.")

check_valid('tf.pad', generated_inputs['tf.pad'], lib="tf", suffix=0)
