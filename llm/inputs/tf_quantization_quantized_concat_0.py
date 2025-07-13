
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantized_concat_inputs():
    list_of_inputs = []

    # Input 1
    concat_dim = np.int32(0)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(5.0)]
    input_maxes = [np.float32(4.0), np.float32(8.0)]
    name = "concat1"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    concat_dim = np.int32(1)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)]
    input_mins = [np.float32(-1.0), np.float32(-5.0)]
    input_maxes = [np.float32(4.0), np.float32(8.0)]
    name = "concat2"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    concat_dim = np.int32(0)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int8)]
    input_mins = [np.float32(1.0), np.float32(9.0)]
    input_maxes = [np.float32(8.0), np.float32(16.0)]
    name = "concat3"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    concat_dim = np.int32(2)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int8)]
    input_mins = [np.float32(-1.0), np.float32(-9.0)]
    input_maxes = [np.float32(8.0), np.float32(16.0)]
    name = "concat4"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    concat_dim = np.int32(0)
    values = [np.array([[1, 2]], dtype=np.int8), np.array([[3, 4]], dtype=np.int8), np.array([[5, 6]], dtype=np.int8)]
    input_mins = [np.float32(1.0), np.float32(3.0), np.float32(5.0)]
    input_maxes = [np.float32(2.0), np.float32(4.0), np.float32(6.0)]
    name = "concat5"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    concat_dim = np.int32(1)
    values = [np.array([[1], [2]], dtype=np.int8), np.array([[3], [4]], dtype=np.int8), np.array([[5], [6]], dtype=np.int8)]
    input_mins = [np.float32(-1.0), np.float32(-3.0), np.float32(-5.0)]
    input_maxes = [np.float32(2.0), np.float32(4.0), np.float32(6.0)]
    name = "concat6"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Different shapes but compatible for concat_dim=0
    concat_dim = np.int32(0)
    values = [np.array([[1, 2, 3]], dtype=np.int8), np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int8)]
    input_mins = [np.float32(1.0), np.float32(4.0)]
    input_maxes = [np.float32(3.0), np.float32(9.0)]
    name = "concat7"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - all negative values
    concat_dim = np.int32(1)
    values = [np.array([[-1, -2], [-3, -4]], dtype=np.int8), np.array([[-5, -6], [-7, -8]], dtype=np.int8)]
    input_mins = [np.float32(-4.0), np.float32(-8.0)]
    input_maxes = [np.float32(-1.0), np.float32(-5.0)]
    name = "concat8"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - 1D tensors
    concat_dim = np.int32(0)
    values = [np.array([1, 2, 3], dtype=np.int8), np.array([4, 5, 6], dtype=np.int8)]
    input_mins = [np.float32(1.0), np.float32(4.0)]
    input_maxes = [np.float32(3.0), np.float32(6.0)]
    name = "concat9"
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - empty name
    concat_dim = np.int32(0)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(5.0)]
    input_maxes = [np.float32(4.0), np.float32(8.0)]
    name = ""
    input_dict = {"concat_dim": concat_dim, "values": values, "input_mins": input_mins, "input_maxes": input_maxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(len(list_of_inputs)):
        input_dict = list_of_inputs[i]
        input_dict["values"] = [np.asarray(v) for v in input_dict["values"]]
        input_dict["input_mins"] = [np.asarray(v).item() for v in input_dict["input_mins"]]
        input_dict["input_maxes"] = [np.asarray(v).item() for v in input_dict["input_maxes"]]
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.quantized_concat"] = tf_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.quantized_concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.quantized_concat'.")

check_valid('tf.quantization.quantized_concat', generated_inputs['tf.quantization.quantized_concat'], lib="tf", suffix=0)
