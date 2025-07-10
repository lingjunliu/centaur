
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_and_crop_jpeg_inputs():
    list_of_inputs = []

    # Input 1
    contents = np.array(tf.io.read_file('image.jpeg')) #Replace 'image.jpeg' with a valid JPEG file.
    crop_window = np.array([0, 0, 100, 100], dtype=np.int32)
    channels = 0
    ratio = 1
    fancy_upscaling = True
    try_recover_truncated = False
    acceptable_fraction = 1.0
    dct_method = ""
    name = None

    input_dict = {
        "contents": contents,
        "crop_window": crop_window,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf_io_decode_and_crop_jpeg_input_list = tf_io_decode_and_crop_jpeg_inputs()
generated_inputs["tf.io.decode_and_crop_jpeg"] = tf_io_decode_and_crop_jpeg_input_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_and_crop_jpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_and_crop_jpeg'.")

check_valid('tf.io.decode_and_crop_jpeg', generated_inputs['tf.io.decode_and_crop_jpeg'], lib="tf", suffix=0)
