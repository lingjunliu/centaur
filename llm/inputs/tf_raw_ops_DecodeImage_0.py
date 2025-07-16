
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decodeimage_inputs():
    list_of_inputs = []

    # Input 3: BMP - Removing problematic BMP data
    #contents = np.array(b'BM\x06\x00\x01\x00\x00\x00\x00\x00\x06\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00', dtype=np.object_)
    #input_dict = {"contents": contents, "channels": 0, "dtype": tf.uint16, "expand_animations": True, "name": None}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: GIF (single frame)
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;', dtype=np.object_)
    input_dict = {"contents": contents, "channels": 0, "dtype": tf.float32, "expand_animations": True, "name": "gif_decode"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: GIF (animated)
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;', dtype=np.object_)
    input_dict = {"contents": contents, "channels": 0, "dtype": tf.uint8, "expand_animations": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No channels specified (default), animated GIF False
    contents = np.array(b'GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;', dtype=np.object_)
    input_dict = {"contents": contents, "channels": 0, "dtype": tf.uint8, "expand_animations": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different number of channels (1) for PNG
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00w4\xde\xaf\x00\x00\x00\nIDATx\xda\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.object_)
    input_dict = {"contents": contents, "channels": 1, "dtype": tf.uint8, "expand_animations": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32 PNG
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xda\xed\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.object_)
    input_dict = {"contents": contents, "channels": 0, "dtype": tf.float32, "expand_animations": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeImage"] = tf_raw_ops_decodeimage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeImage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeImage'.")

check_valid('tf.raw_ops.DecodeImage', generated_inputs['tf.raw_ops.DecodeImage'], lib="tf", suffix=0)
