
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedConcat_inputs():
    list_of_inputs = []

    # Input 1
    concat_dim = np.int32(0)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    concat_dim = np.int32(1)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    concat_dim = np.int32(0)
    values = [np.array([1, 2, 3], dtype=np.uint8), np.array([4, 5, 6], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    concat_dim = np.int32(1)
    values = [np.array([[1, 2, 3]], dtype=np.uint8), np.array([[4, 5, 6]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    concat_dim = np.int32(0)
    values = [np.array([[[1,2],[3,4]]], dtype=np.uint8), np.array([[[5,6],[7,8]]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    concat_dim = np.int32(2)
    values = [np.array([[[1,2],[3,4]]], dtype=np.uint8), np.array([[[5,6],[7,8]]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    concat_dim = np.int32(0)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]

    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    concat_dim = np.int32(0)
    values = [np.array([1, 2], dtype=np.uint8), np.array([3, 4, 5], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    concat_dim = np.int32(0)
    values = [np.array([[[1],[2]],[[3],[4]]], dtype=np.uint8), np.array([[[5],[6]],[[7],[8]]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    concat_dim = np.int32(1)
    values = [np.array([[[1],[2]],[[3],[4]]], dtype=np.uint8), np.array([[[5],[6]],[[7],[8]]], dtype=np.uint8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(10.0), np.float32(10.0)]
    input_dict = {
        "concat_dim": concat_dim,
        "values": values,
        "input_mins": input_mins,
        "input_maxes": input_maxes,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_raw_ops_QuantizedConcat_inputs()
for i, input_dict in enumerate(temp_inputs):
  input_dict['values'] = [tf.convert_to_tensor(x) for x in input_dict['values']]
  input_dict['input_mins'] = [tf.convert_to_tensor(x).numpy() for x in input_dict['input_mins']]
  input_dict['input_maxes'] = [tf.convert_to_tensor(x).numpy() for x in input_dict['input_maxes']]
  input_dict['concat_dim'] = tf.convert_to_tensor(input_dict['concat_dim']).numpy()
  temp_inputs[i] = input_dict
generated_inputs["tf.raw_ops.QuantizedConcat"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConcat'.")

check_valid('tf.raw_ops.QuantizedConcat', generated_inputs['tf.raw_ops.QuantizedConcat'], lib="tf", suffix=0)
