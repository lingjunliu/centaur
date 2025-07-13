
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_add_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(5.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(10.0, dtype=np.float32)
    Toutput = tf.qint32
    name = "add1"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([4, 5, 6], dtype=np.uint8)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(255.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(255.0, dtype=np.float32)
    Toutput = tf.qint32
    name = "add2"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[5, 6], [7, 8]], dtype=np.int16)
    min_x = np.array(-10.0, dtype=np.float32)
    max_x = np.array(10.0, dtype=np.float32)
    min_y = np.array(-5.0, dtype=np.float32)
    max_y = np.array(15.0, dtype=np.float32)
    Toutput = tf.qint32
    name = "add3"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(100.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(150.0, dtype=np.float32)
    Toutput = tf.qint32
    name = "add4"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    min_x = np.array(-100.0, dtype=np.float32)
    max_x = np.array(100.0, dtype=np.float32)
    min_y = np.array(-50.0, dtype=np.float32)
    max_y = np.array(150.0, dtype=np.float32)
    Toutput = tf.qint32
    name = "add5"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(5.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(10.0, dtype=np.float32)
    Toutput = tf.qint8
    name = "add6"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([4, 5, 6], dtype=np.uint8)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(255.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(255.0, dtype=np.float32)
    Toutput = tf.quint8
    name = "add7"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[5, 6], [7, 8]], dtype=np.int16)
    min_x = np.array(-10.0, dtype=np.float32)
    max_x = np.array(10.0, dtype=np.float32)
    min_y = np.array(-5.0, dtype=np.float32)
    max_y = np.array(15.0, dtype=np.float32)
    Toutput = tf.qint16
    name = "add8"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(100.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(150.0, dtype=np.float32)
    Toutput = tf.quint16
    name = "add9"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    min_x = np.array(-100.0, dtype=np.float32)
    max_x = np.array(100.0, dtype=np.float32)
    min_y = np.array(-50.0, dtype=np.float32)
    max_y = np.array(150.0, dtype=np.float32)
    Toutput = tf.qint32
    name = "add10"
    input_dict = {"x": x, "y": y, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "Toutput": Toutput, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_raw_ops_quantized_add_inputs()
processed_inputs = []
for input_dict in temp_inputs:
    processed_input = {}
    processed_input["x"] = tf.convert_to_tensor(input_dict["x"])
    processed_input["y"] = tf.convert_to_tensor(input_dict["y"])
    processed_input["min_x"] = tf.convert_to_tensor(input_dict["min_x"])
    processed_input["max_x"] = tf.convert_to_tensor(input_dict["max_x"])
    processed_input["min_y"] = tf.convert_to_tensor(input_dict["min_y"])
    processed_input["max_y"] = tf.convert_to_tensor(input_dict["max_y"])
    processed_input["Toutput"] = input_dict["Toutput"]
    processed_input["name"] = input_dict["name"]
    processed_inputs.append(processed_input)

generated_inputs["tf.raw_ops.QuantizedAdd"] = processed_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAdd'.")

check_valid('tf.raw_ops.QuantizedAdd', generated_inputs['tf.raw_ops.QuantizedAdd'], lib="tf", suffix=0)
