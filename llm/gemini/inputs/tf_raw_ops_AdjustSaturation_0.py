
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AdjustSaturation_inputs():
    list_of_inputs = []
    
    # Input 1
    images = np.random.rand(2, 2, 3).astype(np.float32)
    scale = np.array(0.5, dtype=np.float32)
    name = "adjust_sat_1"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 2
    images = np.random.rand(1, 3, 3, 3).astype(np.float32)
    scale = np.array(-0.2, dtype=np.float32)
    name = "adjust_sat_2"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 3
    images = np.random.rand(2, 2, 3).astype(np.float32)
    scale = np.array(1.0, dtype=np.float32)
    name = "adjust_sat_3"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 4
    images = np.random.rand(4, 4, 3).astype(np.float32)
    scale = np.array(-1.5, dtype=np.float32)
    name = "adjust_sat_4"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 5
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    scale = np.array(0.0, dtype=np.float32)
    name = "adjust_sat_5"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 6
    images = np.random.rand(5, 5, 5, 3).astype(np.float32)
    scale = np.array(2.5, dtype=np.float32)
    name = "adjust_sat_6"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 7
    images = np.random.rand(1, 1, 3).astype(np.float32)
    scale = np.array(-0.5, dtype=np.float32)
    name = "adjust_sat_7"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 8
    images = np.random.rand(3, 3, 3).astype(np.float32)
    scale = np.array(0.75, dtype=np.float32)
    name = "adjust_sat_8"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 9
    images = np.random.rand(1, 2, 3, 4, 3).astype(np.float32)
    scale = np.array(1.2, dtype=np.float32)
    name = "adjust_sat_9"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 10
    images = np.random.rand(10, 10, 3).astype(np.float32)
    scale = np.array(-2.0, dtype=np.float32)
    name = "adjust_sat_10"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    return list_of_inputs

generated_inputs["tf.raw_ops.AdjustSaturation"] = tf_raw_ops_AdjustSaturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AdjustSaturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AdjustSaturation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AdjustSaturation', generated_inputs['tf.raw_ops.AdjustSaturation'], lib="tf", suffix=0)
