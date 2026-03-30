
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_quantized_concat_inputs():
    list_of_inputs = []

    # Input 1
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[1, 2]], dtype=np.uint8), np.array([[3, 4]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(5.0, dtype=np.float32), np.array(5.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_1",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    concat_dim = np.array(1, dtype=np.int32)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(10.0, dtype=np.float32), np.array(10.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_2",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([1, 2], dtype=np.uint8), np.array([3, 4], dtype=np.uint8), np.array([5, 6], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(255.0, dtype=np.float32), np.array(255.0, dtype=np.float32), np.array(255.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_3",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    concat_dim = np.array(1, dtype=np.int32)
    values = [np.array([[1, 2, 3]], dtype=np.uint8), np.array([[4, 5, 6]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(1.0, dtype=np.float32), np.array(1.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_4",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8), np.array([[9, 10], [11, 12]], dtype=np.uint8)]
    input_mins = [np.array(-1.0, dtype=np.float32), np.array(-1.0, dtype=np.float32), np.array(-1.0, dtype=np.float32)]
    input_maxes = [np.array(1.0, dtype=np.float32), np.array(1.0, dtype=np.float32), np.array(1.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_5",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    concat_dim = np.array(2, dtype=np.int32)
    values = [np.array([[[1, 2], [3, 4]]], dtype=np.uint8), np.array([[[5, 6], [7, 8]]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(255.0, dtype=np.float32), np.array(255.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_6",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([1, 2, 3], dtype=np.uint8), np.array([4, 5, 6], dtype=np.uint8)]
    input_mins = [np.array(-10.0, dtype=np.float32), np.array(-10.0, dtype=np.float32)]
    input_maxes = [np.array(10.0, dtype=np.float32), np.array(10.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_7",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[1, 2, 3]], dtype=np.uint8), np.array([[4, 5, 6]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(20.0, dtype=np.float32), np.array(20.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_8",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    concat_dim = np.array(1, dtype=np.int32)
    values = [np.array([[1], [2]], dtype=np.uint8), np.array([[3], [4]], dtype=np.uint8)]
    input_mins = [np.array(-5.0, dtype=np.float32), np.array(-5.0, dtype=np.float32)]
    input_maxes = [np.array(5.0, dtype=np.float32), np.array(5.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_9",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    concat_dim = np.array(0, dtype=np.int32)
    values = [np.array([[[1, 2], [3, 4]]], dtype=np.uint8), np.array([[[5, 6], [7, 8]]], dtype=np.uint8)]
    input_mins = [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]
    input_maxes = [np.array(15.0, dtype=np.float32), np.array(15.0, dtype=np.float32)]

    input_dict = {
        "name": "test_concat_10",
        "concat_dim": tf.convert_to_tensor(concat_dim, dtype=tf.int32),
        "values": [tf.convert_to_tensor(v, dtype=tf.uint8) for v in values],
        "input_mins": [tf.convert_to_tensor(input_mins[i], dtype=tf.float32) for i in range(len(input_mins))],
        "input_maxes": [tf.convert_to_tensor(input_maxes[i], dtype=tf.float32) for i in range(len(input_maxes))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedConcat"] = tf_raw_ops_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConcat'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedConcat', generated_inputs['tf.raw_ops.QuantizedConcat'], lib="tf", suffix=0)
