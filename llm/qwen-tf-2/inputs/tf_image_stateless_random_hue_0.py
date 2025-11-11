
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 3D tensor with seed
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([1, 2], dtype=np.int32)
    max_delta = 0.2
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with seed
    image = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]], dtype=np.float32)
    seed = np.array([3, 4], dtype=np.int32)
    max_delta = 0.1
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Zero delta value (valid)
    image = np.array([[[0.5, 0.4, 0.3], [0.2, 0.1, 0.0]], [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)
    seed = np.array([5, 6], dtype=np.int32)
    max_delta = 0.0
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Large delta value (valid)
    image = np.array([[[0.5, 0.4, 0.3], [0.2, 0.1, 0.0]], [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)
    seed = np.array([7, 8], dtype=np.int32)
    max_delta = 0.4
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single pixel image
    image = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    seed = np.array([9, 10], dtype=np.int32)
    max_delta = 0.2
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different color channels (RGB)
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([11, 12], dtype=np.int32)
    max_delta = 0.5
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large number of channels
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([13, 14], dtype=np.int32)
    max_delta = 0.4
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Float values with different dimensions
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([15, 16], dtype=np.int32)
    max_delta = 0.3
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large delta value with different seed
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([17, 18], dtype=np.int32)
    max_delta = 0.5
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Another valid case with zero delta
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([19, 20], dtype=np.int32)
    max_delta = 0.0
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_hue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_hue', generated_inputs['tf.image.stateless_random_hue'], lib="tf", suffix=0)
