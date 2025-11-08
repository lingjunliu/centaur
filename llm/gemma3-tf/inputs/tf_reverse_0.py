
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_reverse_inputs():
    list_of_inputs = []

    tensor1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis1 = np.array([0], dtype=np.int32)
    name1 = "reverse_1"
    input_dict1 = {"tensor": tensor1, "axis": axis1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    tensor2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis2 = np.array([1], dtype=np.int32)
    name2 = "reverse_2"
    input_dict2 = {"tensor": tensor2, "axis": axis2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    tensor3 = np.array([1, 2, 3, 4], dtype=np.int32)
    axis3 = np.array([0], dtype=np.int32)
    name3 = "reverse_3"
    input_dict3 = {"tensor": tensor3, "axis": axis3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    tensor5 = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]], dtype=np.int32)
    axis5 = np.array([-1], dtype=np.int32)
    name5 = "reverse_5"
    input_dict5 = {"tensor": tensor5, "axis": axis5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    tensor6 = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]], dtype=np.int32)
    axis6 = np.array([0, 2], dtype=np.int32)
    name6 = "reverse_6"
    input_dict6 = {"tensor": tensor6, "axis": axis6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    tensor7 = np.array([1, 2, 3], dtype=np.float32)
    axis7 = np.array([0], dtype=np.int32)
    name7 = "reverse_7"
    input_dict7 = {"tensor": tensor7, "axis": axis7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    tensor8 = np.array([[True, False], [False, True]], dtype=np.bool_)
    axis8 = np.array([1], dtype=np.int32)
    name8 = "reverse_8"
    input_dict8 = {"tensor": tensor8, "axis": axis8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    tensor9 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis9 = np.array([0, 1], dtype=np.int64)
    name9 = "reverse_9"
    input_dict9 = {"tensor": tensor9, "axis": axis9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse'], lib="tf", suffix=0)
