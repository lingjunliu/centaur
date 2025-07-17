
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_euclidean_norm_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis1 = np.array([0], dtype=np.int32)
    keep_dims1 = False
    name1 = "euclidean_norm_1"
    input_dict1 = {"input": input1, "axis": axis1, "keep_dims": keep_dims1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis2 = np.array([1], dtype=np.int32)
    keep_dims2 = True
    name2 = "euclidean_norm_2"
    input_dict2 = {"input": input2, "axis": axis2, "keep_dims": keep_dims2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    axis3 = np.array([0, 1], dtype=np.int64)
    keep_dims3 = False
    name3 = "euclidean_norm_3"
    input_dict3 = {"input": input3, "axis": axis3, "keep_dims": keep_dims3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis4 = np.array([0], dtype=np.int32)
    keep_dims4 = True
    name4 = "euclidean_norm_4"
    input_dict4 = {"input": input4, "axis": axis4, "keep_dims": keep_dims4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([[-1, -2], [-3, -4]], dtype=np.int16)
    axis5 = np.array([0], dtype=np.int32)
    keep_dims5 = False
    name5 = "euclidean_norm_5"
    input_dict5 = {"input": input5, "axis": axis5, "keep_dims": keep_dims5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.array([[1j, 2j], [3j, 4j]], dtype=np.complex64)
    axis6 = np.array([1], dtype=np.int32)
    keep_dims6 = True
    name6 = "euclidean_norm_6"
    input_dict6 = {"input": input6, "axis": axis6, "keep_dims": keep_dims6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis7 = np.array([0, 2], dtype=np.int64)
    keep_dims7 = True
    name7 = "euclidean_norm_7"
    input_dict7 = {"input": input7, "axis": axis7, "keep_dims": keep_dims7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis8 = np.array([0], dtype=np.int32)
    keep_dims8 = False
    name8 = "euclidean_norm_8"
    input_dict8 = {"input": input8, "axis": axis8, "keep_dims": keep_dims8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    axis9 = np.array([-1], dtype=np.int32)
    keep_dims9 = False
    name9 = "euclidean_norm_9"
    input_dict9 = {"input": input9, "axis": axis9, "keep_dims": keep_dims9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.array([1, 2, 3, 4], dtype=np.uint8)
    axis10 = np.array([0], dtype=np.int32)
    keep_dims10 = False
    name10 = "euclidean_norm_10"
    input_dict10 = {"input": input10, "axis": axis10, "keep_dims": keep_dims10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EuclideanNorm"] = tf_raw_ops_euclidean_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EuclideanNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EuclideanNorm'.")

check_valid('tf.raw_ops.EuclideanNorm', generated_inputs['tf.raw_ops.EuclideanNorm'], lib="tf", suffix=0)
