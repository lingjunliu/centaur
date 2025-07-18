
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# This function will generate valid inputs for tf.raw_ops.DecodeGif.
# The previous attempts failed due to invalid image data for PNG and JPEG formats when
# used with this specific op. This version focuses exclusively on providing known-valid,
# simple, uncompressed GIF byte strings, which is the primary intended use case for this op.

# A minimal 1x1 black GIF.
MINIMAL_BLACK_GIF_BYTES = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

# A minimal 1x1 white GIF.
MINIMAL_WHITE_GIF_BYTES = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00;'

# A minimal 1x1 red GIF.
MINIMAL_RED_GIF_BYTES = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

# A minimal 1x1 green GIF.
MINIMAL_GREEN_GIF_BYTES = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\xff\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

# A minimal 1x1 blue GIF.
MINIMAL_BLUE_GIF_BYTES = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\xff\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'


def tf_raw_ops_decode_gif_inputs():
    list_of_inputs = []

    # Case 1: Black GIF with a simple name
    input_dict = {
        'contents': np.array(MINIMAL_BLACK_GIF_BYTES, dtype=np.string_),
        'name': 'decode_black_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: White GIF with a simple name
    input_dict = {
        'contents': np.array(MINIMAL_WHITE_GIF_BYTES, dtype=np.string_),
        'name': 'decode_white_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Red GIF with a simple name
    input_dict = {
        'contents': np.array(MINIMAL_RED_GIF_BYTES, dtype=np.string_),
        'name': 'decode_red_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Green GIF with a different name
    input_dict = {
        'contents': np.array(MINIMAL_GREEN_GIF_BYTES, dtype=np.string_),
        'name': 'green_gif_decoder'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Blue GIF with a different name
    input_dict = {
        'contents': np.array(MINIMAL_BLUE_GIF_BYTES, dtype=np.string_),
        'name': 'MyBlueGifDecoder'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Black GIF with an empty string name
    input_dict = {
        'contents': np.array(MINIMAL_BLACK_GIF_BYTES, dtype=np.string_),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: White GIF with a long name
    input_dict = {
        'contents': np.array(MINIMAL_WHITE_GIF_BYTES, dtype=np.string_),
        'name': 'a_very_long_and_descriptive_name_for_this_white_gif_decoding_operation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Red GIF with a name containing special characters
    input_dict = {
        'contents': np.array(MINIMAL_RED_GIF_BYTES, dtype=np.string_),
        'name': 'decode_gif_op/node_v1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Green GIF with an empty string name
    input_dict = {
        'contents': np.array(MINIMAL_GREEN_GIF_BYTES, dtype=np.string_),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Blue GIF with a different name style
    input_dict = {
        'contents': np.array(MINIMAL_BLUE_GIF_BYTES, dtype=np.string_),
        'name': 'blue_decoder_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 11: Black GIF with another different name
    input_dict = {
        'contents': np.array(MINIMAL_BLACK_GIF_BYTES, dtype=np.string_),
        'name': 'black_gif_op_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeGif"] = tf_raw_ops_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeGif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeGif'.")

check_valid('tf.raw_ops.DecodeGif', generated_inputs['tf.raw_ops.DecodeGif'], lib="tf", suffix=0)
