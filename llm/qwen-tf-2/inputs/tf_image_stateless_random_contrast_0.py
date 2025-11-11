
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: 3D image tensor with seed
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([1, 2], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.2,
        "upper": 0.5,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D image tensor with seed
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                     [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]], dtype=np.float32)
    seed = np.array([3, 4], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.1,
        "upper": 0.9,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D image tensor with negative lower bound
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    seed = np.array([5, 6], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": -0.5,
        "upper": 0.5,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D image tensor with seed and upper bound greater than lower
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([7, 8], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.3,
        "upper": 0.9,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D image tensor with negative values
    image = np.array([[[[1.0, -2.0], [-3.0, 4.0]], [[5.0, -6.0], [-7.0, 8.0]]], 
                     [[[9.0, -10.0], [-11.0, 12.0]], [[13.0, -14.0], [-15.0, 16.0]]]], dtype=np.float32)
    seed = np.array([9, 10], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": -0.3,
        "upper": 0.8,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D image tensor with seed and different values
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([11, 12], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.2,
        "upper": 0.8,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D image tensor with seed and lower bound higher than upper
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    seed = np.array([13, 14], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.9,
        "upper": 0.3,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D image tensor with different dimensions and seed
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                     [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]], dtype=np.float32)
    seed = np.array([15, 16], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.1,
        "upper": 0.9,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D image tensor with different dimensions and seed
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                     [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
                     [[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]]], dtype=np.float32)
    seed = np.array([17, 18], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.2,
        "upper": 0.8,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D image tensor with seed and zero lower bound
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([19, 20], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "lower": 0.0,
        "upper": 0.5,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast"] = generate_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_contrast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_contrast', generated_inputs['tf.image.stateless_random_contrast'], lib="tf", suffix=0)
