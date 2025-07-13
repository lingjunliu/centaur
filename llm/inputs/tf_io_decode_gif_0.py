
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid GIF (empty). May throw error when executed since the content is not a valid gif file.
    # Removed this as it causes errors
    # contents = np.array(b"")
    # name = "gif1"
    # input_dict = {"contents": contents, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Slightly longer, but still invalid gif
    # Removed since invalid GIF data.
    # contents = np.array(b"GIF89a")  # GIF header. May throw error when executed since the content is not a valid gif file.
    # name = "gif2"
    # input_dict = {"contents": contents, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty name
    contents = np.array(b"GIF89a") # Invalid gif but string content
    name = ""
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Long name
    contents = np.array(b"GIF89a") # Invalid gif but string content
    name = "a_very_long_name_for_the_gif_image"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Name with special characters
    contents = np.array(b"GIF89a") # Invalid gif but string content
    name = "gif_with_!@#$%^&*()_+=-`~[]{}|;':\",./<>?"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Name with numbers
    contents = np.array(b"GIF89a") # Invalid gif but string content
    name = "gif12345"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Name with unicode characters
    contents = np.array(b"GIF89a") # Invalid gif but string content
    name = "gif_你好世界"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Short GIF header
    contents = np.array(b"GIF") # Invalid gif but string content
    name = "short_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name with spaces.
    contents = np.array(b"GIF89a")  # Invalid gif but string content
    name = "gif with spaces"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Another invalid gif content.
    contents = np.array(b"Invalid GIF Content")
    name = "invalid_gif"
    input_dict = {"contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_gif"] = tf_io_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_gif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_gif'.")

check_valid('tf.io.decode_gif', generated_inputs['tf.io.decode_gif'], lib="tf", suffix=0)
