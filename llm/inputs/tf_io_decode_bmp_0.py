
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input
    contents = np.array(b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype='uint8')
    channels = 0
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specify channels = 3
    contents = np.array(b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype='uint8')
    channels = 3
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specify channels = 4
    contents = np.array(b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype='uint8')
    channels = 4
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: with name
    contents = np.array(b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype='uint8')
    channels = 0
    name = "bmp_decode_test"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different bmp data
    contents = np.array(b'BMZ\x0c\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x80\x0b\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00', dtype='uint8')
    channels = 0
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Channels 3 with different bmp data
    contents = np.array(b'BMZ\x0c\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x80\x0b\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00', dtype='uint8')
    channels = 3
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Channels 4 with different bmp data
    contents = np.array(b'BMZ\x0c\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x80\x0b\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00', dtype='uint8')
    channels = 4
    name = None
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Name and different bmp data
    contents = np.array(b'BMZ\x0c\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x80\x0b\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00', dtype='uint8')
    channels = 0
    name = "different_bmp"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Minimal bmp with different name
    contents = np.array(b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype='uint8')
    channels = 0
    name = "minimal_bmp"
    input_dict = {"contents": contents.tobytes(), "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different name, channels = 3
    contents = np.array(b'BMZ\x0c\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x80\x0b\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00', dtype='uint8')
    channels = 3
    name = "diff_name_chan3"
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
