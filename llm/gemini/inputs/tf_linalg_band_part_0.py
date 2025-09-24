
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_band_part_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    num_lower = np.int32(1)
    num_upper = np.int32(1)
    name = "band_part_1"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    num_lower = np.int32(-1)
    num_upper = np.int32(0)
    name = "band_part_2"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int64)
    num_lower = np.int64(0)
    num_upper = np.int64(-1)
    name = "band_part_3"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float64)
    num_lower = np.int32(0)
    num_upper = np.int32(0)
    name = "band_part_4"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    num_lower = np.int32(2)
    num_upper = np.int32(1)
    name = "band_part_5"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    num_lower = np.int32(1)
    num_upper = np.int32(-1)
    name = "band_part_6"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    num_lower = np.int32(-1)
    num_upper = np.int32(1)
    name = "band_part_7"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    num_lower = np.int32(0)
    num_upper = np.int32(0)
    name = "band_part_8"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]).astype(np.int32)
    num_lower = np.int32(1)
    num_upper = np.int32(0)
    name = "band_part_9"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]).astype(np.float64)
    num_lower = np.int32(0)
    num_upper = np.int32(1)
    name = "band_part_10"
    input_dict = {"input": input_tensor, "num_lower": num_lower, "num_upper": num_upper, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.band_part"] = tf_linalg_band_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.band_part' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.band_part'.")

check_valid('tf.linalg.band_part', generated_inputs['tf.linalg.band_part'], lib="tf", suffix=0)
