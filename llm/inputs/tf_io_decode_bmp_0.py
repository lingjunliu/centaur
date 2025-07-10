
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input, empty BMP
    contents = np.array(b"\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    channels = 0
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specify channels=3
    contents = np.array(b"\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    channels = 3
    name = "decode_rgb"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specify channels=4
    contents = np.array(b"\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    channels = 4
    name = "decode_rgba"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different name
    contents = np.array(b"\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    channels = 0
    name = "another_name"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another valid BMP with actual pixel data (1x1 RGB)
    contents = np.array(b"\x42\x4D\x38\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x18\x00\x18\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\x00\x00")
    channels = 0
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different pixel value (1x1 RGB)
    contents = np.array(b"\x42\x4D\x38\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x18\x00\x18\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\x00")
    channels = 0
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: 2x1 RGB BMP
    contents = np.array(b"\x42\x4D\x44\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x18\x00\x18\x00\x00\x00\x00\x00\x0E\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\x00\x00\x00\xFF\x00")
    channels = 0
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small 1x1 bmp, channels=3, name="bmp_image"
    contents = np.array(b"\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    channels = 3
    name = "bmp_image"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small 1x1 bmp, channels=4
    contents = np.array(b"\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    channels = 4
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: BMP with palette.
    contents = np.array(b'\x42\x4d\x76\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x08\x00\x08\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x80\x00\x00\x80\x80\x00\x00\x80\x00\x80\x00\x80\x80\x00\x80\x00\x00\x80\x00\x80\x00\x80\x80\x00\x80\x80\x80\x00\xff\x00\x00\x00\x00\x00')
    channels = 0
    name = "palette_bmp"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
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
