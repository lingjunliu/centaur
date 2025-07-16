
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Valid GIF (minimal valid)
    try:
      with open("valid_gif.gif", "rb") as f:
        valid_gif_data = f.read()
    except:
      valid_gif_data = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

    input_dict = {
        "contents": np.array(valid_gif_data, dtype=np.string_),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid GIF with a name
    try:
      with open("valid_gif.gif", "rb") as f:
        valid_gif_data = f.read()
    except:
      valid_gif_data = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

    input_dict = {
        "contents": np.array(valid_gif_data, dtype=np.string_),
        "name": "gif_with_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid gif with a comment
    try:
      with open("valid_gif.gif", "rb") as f:
        valid_gif_data = f.read()
    except:
      valid_gif_data = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

    input_dict = {
        "contents": np.array(valid_gif_data + b'!\xfe\x0bCOMMENT EXT;\x00', dtype=np.string_),
        "name": "gif_with_comment"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid gif with application extension
    try:
      with open("valid_gif.gif", "rb") as f:
        valid_gif_data = f.read()
    except:
      valid_gif_data = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    input_dict = {
        "contents": np.array(valid_gif_data + b'!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00', dtype=np.string_),
        "name": "gif_with_application_extension"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid gif - more complex
    try:
        with open("valid_gif3.gif", "rb") as f:
            valid_gif_data3 = f.read()
    except:
        valid_gif_data3 = b'GIF89a\x10\x00\x10\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x10\x00\x10\x00\x00\x02\x1fZ\xc1\xdc\x8b\xfa\x00\x91\x97\x0b\x0c\x05\xa3\x01\x86\xfa\xc8\x95q\x92\x98L\xc7\x00;\x00'

    input_dict = {
        "contents": np.array(valid_gif_data3, dtype=np.string_),
        "name": "complex_gif"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: GIF with interlacing
    try:
        with open("valid_gif5.gif", "rb") as f:
            valid_gif_data5 = f.read()
    except:
        valid_gif_data5 = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x40\x02\x02D\x01\x00;'

    input_dict = {
        "contents": np.array(valid_gif_data5, dtype=np.string_),
        "name": "interlaced_gif"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Valid GIF as bytes object directly
    try:
      with open("valid_gif.gif", "rb") as f:
        valid_gif_data = f.read()
    except:
      valid_gif_data = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    input_dict = {
        "contents": np.array(valid_gif_data, dtype=np.string_),
        "name": "bytes_input_gif"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Valid GIF - Different resolution
    try:
        with open("valid_gif6.gif", "rb") as f:
            valid_gif_data6 = f.read()
    except:
        valid_gif_data6 = b'GIF89a\x02\x00\x02\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02\x02D\x01\x00;'
    input_dict = {
        "contents": np.array(valid_gif_data6, dtype=np.string_),
        "name": "diff_resolution_gif"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty name
    try:
      with open("valid_gif.gif", "rb") as f:
        valid_gif_data = f.read()
    except:
      valid_gif_data = b'GIF89a\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    input_dict = {
        "contents": np.array(valid_gif_data, dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 : adding a new valid GIF - Different but valid structure
    try:
        with open("valid_gif8.gif", "rb") as f:
            valid_gif_data8 = f.read()
    except:
        valid_gif_data8 = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x00\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00;'

    input_dict = {
        "contents": np.array(valid_gif_data8, dtype=np.string_),
        "name": "valid_gif_8"
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeGif"] = tf_raw_ops_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeGif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeGif'.")

check_valid('tf.raw_ops.DecodeGif', generated_inputs['tf.raw_ops.DecodeGif'], lib="tf", suffix=0)
