
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixBandPart_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower_tensor = np.array(1, dtype=np.int32)
    num_upper_tensor = np.array(1, dtype=np.int32)
    name = "band_part_1"
    input_dict = {"input": input_tensor, "num_lower": num_lower_tensor, "num_upper": num_upper_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower_tensor = np.array(-1, dtype=np.int32)
    num_upper_tensor = np.array(-1, dtype=np.int32)
    name = "band_part_2"
    input_dict = {"input": input_tensor, "num_lower": num_lower_tensor, "num_upper": num_upper_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower_tensor = np.array(0, dtype=np.int32)
    num_upper_tensor = np.array(0, dtype=np.int32)
    name = "band_part_3"
    input_dict = {"input": input_tensor, "num_lower": num_lower_tensor, "num_upper": num_upper_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower_tensor = np.array(1, dtype=np.int32)
    num_upper_tensor = np.array(-1, dtype=np.int32)
    name = "band_part_4"
    input_dict = {"input": input_tensor, "num_lower": num_lower_tensor, "num_upper": num_upper_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    num_lower_tensor = np.array(-1, dtype=np.int32)
    num_upper_tensor = np.array(0, dtype=np.int32)
    name = "band_part_5"
    input_dict = {"input": input_tensor, "num_lower": num_lower_tensor, "num_upper": num_upper_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_raw_ops_MatrixBandPart_inputs()
for input_dict in temp_inputs:
  input_dict["input"] = tf.convert_to_tensor(input_dict["input"])
  input_dict["num_lower"] = tf.convert_to_tensor(input_dict["num_lower"])
  input_dict["num_upper"] = tf.convert_to_tensor(input_dict["num_upper"])

generated_inputs["tf.raw_ops.MatrixBandPart"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixBandPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixBandPart'.")

check_valid('tf.raw_ops.MatrixBandPart', generated_inputs['tf.raw_ops.MatrixBandPart'], lib="tf", suffix=0)
