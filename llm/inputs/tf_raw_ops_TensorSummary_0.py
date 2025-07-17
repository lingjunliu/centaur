
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_TensorSummary_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([1, 2, 3], dtype=np.int32)
    description = ""
    labels = []
    display_name = ""
    name = "summary_1"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    description = "A 2x2 matrix"
    labels = []
    display_name = "Matrix A"
    name = "summary_2"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([True, False, True], dtype=np.bool_)
    description = "Boolean array"
    labels = []
    display_name = "Boolean Values"
    name = "summary_3"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array(["a", "b", "c"], dtype=np.unicode_)
    description = "String array"
    labels = []
    display_name = "String Data"
    name = "summary_4"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([-1, -2, -3], dtype=np.int64)
    description = "Negative integers"
    labels = []
    display_name = "Negative Ints"
    name = "summary_5"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array(1.2345, dtype=np.float64)
    description = "Single float"
    labels = []
    display_name = "Single Float"
    name = "summary_6"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([1, 2, 3], dtype=np.uint8)
    description = "Unsigned int8"
    labels = []
    display_name = "Unsigned Int8"
    name = "summary_7"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([1, 2, 3], dtype=np.int16)
    description = "Int16"
    labels = []
    display_name = "Int16"
    name = "summary_8"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    description = "3D tensor"
    labels = []
    display_name = "3D Tensor"
    name = "summary_9"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([], dtype=np.float32)
    description = "Empty tensor"
    labels = []
    display_name = "Empty Tensor"
    name = "summary_10"
    input_dict = {
        "tensor": tensor,
        "description": description,
        "labels": labels,
        "display_name": display_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TensorSummary"] = tf_raw_ops_TensorSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TensorSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TensorSummary'.")

check_valid('tf.raw_ops.TensorSummary', generated_inputs['tf.raw_ops.TensorSummary'], lib="tf", suffix=0)
