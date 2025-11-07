
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_grayscale_to_rgb_inputs():
    list_of_inputs = []
    
    # Input 1: Single channel image (3D)
    a = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Single channel image with negative values (3D)
    a = np.array([[[ -1.0], [0.0], [1.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Multi-channel image (4D)
    a = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Multi-channel image with negative values (4D)
    a = np.array([[[[ -1.0], [0.0]], [[1.0], [2.0]]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single channel image with float values (3D)
    a = np.array([[[1.5], [2.7], [3.9]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single channel image with mixed values (3D)
    a = np.array([[[0.0], [1.0], [2.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single channel image with zero values (3D)
    a = np.array([[[0.0], [0.0], [0.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single channel image with large values (3D)
    a = np.array([[[100.0], [200.0], [300.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single channel image with decimal values (3D)
    a = np.array([[[1.1], [2.2], [3.3]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Single channel image with negative values (3D)
    a = np.array([[[ -5.0], [ -10.0], [ -15.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = generate_grayscale_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.grayscale_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.grayscale_to_rgb'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.grayscale_to_rgb', generated_inputs['tf.image.grayscale_to_rgb'], lib="tf", suffix=0)
