
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantizedownandshrinkrange_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_min = np.array(-1.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    out_type = tf.qint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 2
    input_tensor = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int8)
    input_min = np.array(-2.0, dtype=np.float32)
    input_max = np.array(4.0, dtype=np.float32)
    out_type = tf.quint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 3
    input_tensor = np.array([100, 200, 300], dtype=np.int16)
    input_min = np.array(50.0, dtype=np.float32)
    input_max = np.array(350.0, dtype=np.float32)
    out_type = tf.quint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 4
    input_tensor = np.array([[-1000, -500], [0, 500]], dtype=np.int32)
    input_min = np.array(-1200.0, dtype=np.float32)
    input_max = np.array(600.0, dtype=np.float32)
    out_type = tf.qint16
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 5
    input_tensor = np.array([65000], dtype=np.uint16)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(70000.0, dtype=np.float32)
    out_type = tf.quint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 6
    input_tensor = np.array([[-128, 0, 127]], dtype=np.int8)
    input_min = np.array(-128.0, dtype=np.float32)
    input_max = np.array(127.0, dtype=np.float32)
    out_type = tf.quint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 7
    input_tensor = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(255.0, dtype=np.float32)
    out_type = tf.qint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 8
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int16)
    input_min = np.array(-10.0, dtype=np.float32)
    input_max = np.array(10.0, dtype=np.float32)
    out_type = tf.qint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 9
    input_tensor = np.array([10, 20, 30, 40], dtype=np.uint16)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(65535.0, dtype=np.float32)
    out_type = tf.quint8
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    # Input 10
    input_tensor = np.array([[-100, 0, 100]], dtype=np.int32)
    input_min = np.array(-200.0, dtype=np.float32)
    input_max = np.array(200.0, dtype=np.float32)
    out_type = tf.qint16
    input_dict = {"input": input_tensor, "input_min": input_min, "input_max": input_max, "out_type": out_type, "name": None}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeDownAndShrinkRange"] = []
temp_inputs = tf_raw_ops_quantizedownandshrinkrange_inputs()
for input_dict in temp_inputs:
    generated_inputs["tf.raw_ops.QuantizeDownAndShrinkRange"].append({"kwargs": input_dict})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeDownAndShrinkRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeDownAndShrinkRange'.")

check_valid('tf.raw_ops.QuantizeDownAndShrinkRange', generated_inputs['tf.raw_ops.QuantizeDownAndShrinkRange'], lib="tf", suffix=0)
