
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input (32 bytes)
    contents = np.array(b'\x42\x4D\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x08\x00\x00\x00', dtype=np.string_)
    channels = 0
    name = None
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specify channels = 3 (32 bytes)
    contents = np.array(b'\x42\x4D\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x08\x00\x00\x00', dtype=np.string_)
    channels = 3
    name = "decode_bmp_3_channels"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specify channels = 4 (32 bytes)
    contents = np.array(b'\x42\x4D\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x08\x00\x00\x00', dtype=np.string_)
    channels = 4
    name = "decode_bmp_4_channels"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different name (32 bytes)
    contents = np.array(b'\x42\x4D\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x08\x00\x00\x00', dtype=np.string_)
    channels = 0
    name = "another_name"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer BMP content (more than 32 bytes)
    contents = np.array(b'\x42\x4D\x46\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    channels = 0
    name = None
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: another valid BMP with more data
    contents = np.array(b'\x42\x4d\x46\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    channels = 0
    name = "bmp_6"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: another valid BMP, channels = 3 with more data
    contents = np.array(b'\x42\x4d\x46\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    channels = 3
    name = "bmp_7"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: another valid BMP, channels = 4 with more data
    contents = np.array(b'\x42\x4d\x46\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    channels = 4
    name = "bmp_8"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different name with more data
    contents = np.array(b'\x42\x4d\x46\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    channels = 0
    name = "name_test3"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different BMP content with color data and more data
    contents = np.array(b'\x42\x4D\x7E\x00\x00\x00\x00\x00\x00\x00\x76\x00\x00\x00\x28\x00\x00\x00\x0A\x00\x00\x00\x0A\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x08\x00\x00\x00\x13\x0B\x00\x00\x13\x0B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\x00\x00\x00\x00\xFF\x00\x00\x00\x00\xFF\x00\xFF\x00\x00\x00\x00\x00\xFF\x00\x00\x00\xFF\xFF\x00\x00\xFF\x00\x00\x00\xFF\xFF\x00\xFF\x00\x00\x00\x00\xFF\xFF\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    channels = 0
    name = None
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_bmp"] = tf_io_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_bmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_bmp'.")

check_valid('tf.io.decode_bmp', generated_inputs['tf.io.decode_bmp'], lib="tf", suffix=0)
