
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
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    description = "A 2x2 float matrix"
    labels = ["row1", "row2"]
    display_name = "Matrix A"
    name = "matrix_a_summary"
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([True, False, True], dtype=np.bool_)
    description = "Boolean array"
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array(["hello", "world"], dtype=np.string_)
    description = "String array"
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([1, 2, 3], dtype=np.int64)
    description = ""
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array(-1, dtype=np.int32)
    description = "Negative scalar"
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    description = "Float array"
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    description = "3D int array"
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([], dtype=np.int32)
    description = "Empty array"
    labels = []
    display_name = ""
    name = ""
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array(1, dtype=np.int32)
    description = "Scalar Value"
    labels = []
    display_name = "Scalar"
    name = "scalar_summary"
    input_dict = {"tensor": tensor, "description": description, "labels": labels, "display_name": display_name, "name": name}
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
