
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_extract_patches_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.rand(1, 10, 10, 1).astype(np.float32)
    sizes = [1, 3, 3, 1]
    strides = [1, 5, 5, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = None

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.random.rand(1, 20, 20, 3).astype(np.float32)
    sizes = [1, 5, 5, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 2, 2, 1]
    padding = 'SAME'
    name = "extract_patches_2"

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.random.rand(2, 15, 15, 1).astype(np.float32)
    sizes = [1, 4, 4, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = None

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    images = np.random.rand(1, 12, 12, 5).astype(np.float32)
    sizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 3, 3, 1]
    padding = 'SAME'
    name = "extract_patches_4"

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    images = np.random.rand(1, 8, 8, 1).astype(np.float32)
    sizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = None

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    sizes = [1, 2, 2, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 2, 2, 1]
    padding = 'SAME'
    name = "extract_patches_6"

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    images = np.random.rand(1, 16, 16, 1).astype(np.float32)
    sizes = [1, 5, 5, 1]
    strides = [1, 4, 4, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = None

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = np.random.rand(1, 24, 24, 5).astype(np.float32)
    sizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 3, 3, 1]
    padding = 'SAME'
    name = "extract_patches_8"

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.random.rand(3, 10, 10, 2).astype(np.float32)
    sizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = None

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    images = np.random.rand(1, 14, 14, 4).astype(np.float32)
    sizes = [1, 4, 4, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 2, 2, 1]
    padding = 'SAME'
    name = "extract_patches_10"

    input_dict = {
        "images": tf.convert_to_tensor(images).numpy(),
        "sizes": sizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.extract_patches"] = tf_image_extract_patches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.extract_patches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.extract_patches'.")

check_valid('tf.image.extract_patches', generated_inputs['tf.image.extract_patches'], lib="tf", suffix=0)
