
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fake_quant_with_min_max_vars_per_channel_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 1.5]], dtype=np.float32)
    min_val = np.array([-2.0, -0.5, 0.5], dtype=np.float32)
    max_val = np.array([2.0, 1.0, 2.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test1"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([[-0.5, 0.0, 0.5], [-1.0, 0.25, 0.75]], dtype=np.float32)
    min_val = np.array([-1.0, -0.25, 0.25], dtype=np.float32)
    max_val = np.array([1.0, 0.75, 1.0], dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "test2"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    min_val = np.array([0.0, 0.1], dtype=np.float32)
    max_val = np.array([1.0, 1.1], dtype=np.float32)
    num_bits = 16
    narrow_range = False
    name = "test3"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    max_val = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 2
    narrow_range = True
    name = "test4"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([[-1.0, -0.5], [0.0, 0.5]], dtype=np.float32)
    min_val = np.array([-1.5, -0.75], dtype=np.float32)
    max_val = np.array([0.5, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test5"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = tf_raw_ops_fake_quant_with_min_max_vars_per_channel_inputs()
for i in generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"]:
    i["inputs"] = tf.convert_to_tensor(i["inputs"], dtype=tf.float32)
    i["min"] = tf.convert_to_tensor(i["min"], dtype=tf.float32)
    i["max"] = tf.convert_to_tensor(i["max"], dtype=tf.float32)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'], lib="tf", suffix=0)
