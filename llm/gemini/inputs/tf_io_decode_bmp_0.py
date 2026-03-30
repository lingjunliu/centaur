
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input with 0 channels
    contents = np.array(b'\x42\x4d\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00', dtype=np.string_)
    channels = 0
    name = None
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: RGB channels
    contents = np.array(b'\x42\x4d\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00', dtype=np.string_)
    channels = 3
    name = "rgb_image"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGBA channels
    contents = np.array(b'\x42\x4d\x3e\x00\x00\x00\x00\x00\x00\x00\x3e\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x20\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff', dtype=np.string_)
    channels = 4
    name = "rgba_image"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_bmp"] = tf_io_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_bmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_bmp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.decode_bmp', generated_inputs['tf.io.decode_bmp'], lib="tf", suffix=0)
