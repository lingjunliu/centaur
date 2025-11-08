
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: 3D image with crop
    image = np.arange(27).reshape(3, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 3,
        "target_width": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D image with pad
    image = np.arange(9).reshape(3, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 5,
        "target_width": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D image with crop
    image = np.arange(100).reshape(5, 5, 4, 5)
    input_dict = {
        "image": image,
        "target_height": 2,
        "target_width": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D image with pad
    image = np.arange(9).reshape(3, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 3,
        "target_width": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D image with crop
    image = np.arange(27).reshape(3, 3, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 1,
        "target_width": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D image with pad
    image = np.arange(27).reshape(3, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 4,
        "target_width": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D image with pad
    image = np.arange(16).reshape(2, 2, 4, 2)
    input_dict = {
        "image": image,
        "target_height": 4,
        "target_width": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D image with crop
    image = np.arange(12).reshape(4, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 2,
        "target_width": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D image with crop
    image = np.arange(144).reshape(6, 6, 3, 4)
    input_dict = {
        "image": image,
        "target_height": 4,
        "target_width": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D image with pad
    image = np.arange(27).reshape(3, 3, 3)
    input_dict = {
        "image": image,
        "target_height": 6,
        "target_width": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.resize_with_crop_or_pad"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.resize_with_crop_or_pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize_with_crop_or_pad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.resize_with_crop_or_pad', generated_inputs['tf.image.resize_with_crop_or_pad'], lib="tf", suffix=0)
