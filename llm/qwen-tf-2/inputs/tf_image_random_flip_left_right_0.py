
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (HWC format)
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = 5
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (BHWC format)
    images = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.int32)
    seed = 6
    input_dict = {
        "image": images,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with float values
    image = np.array([[[1.5], [2.7]], [[3.1], [4.9]]], dtype=np.float32)
    seed = 7
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with negative values
    images = np.array([[[[-1], [-2]], [[-3], [-4]]], [[[-5], [-6]], [[-7], [-8]]]], dtype=np.int32)
    seed = 8
    input_dict = {
        "image": images,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with mixed values
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    seed = 9
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with single channel
    images = np.array([[[[1]], [[2]]], [[[3]], [[4]]], dtype=np.int32)
    seed = 10
    input_dict = {
        "image": images,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with high value integers
    image = np.array([[[100], [200]], [[300], [400]]], dtype=np.int32)
    seed = 11
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with different channels
    images = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    seed = 12
    input_dict = {
        "image": images,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with zero values
    image = np.array([[[0], [1]], [[2], [3]]], dtype=np.int32)
    seed = 13
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with large number of channels
    images = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]], dtype=np.int32)
    seed = 14
    input_dict = {
        "image": images,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_flip_left_right', generated_inputs['tf.image.random_flip_left_right'], lib="tf", suffix=0)
