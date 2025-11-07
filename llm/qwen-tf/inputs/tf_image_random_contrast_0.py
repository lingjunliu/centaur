
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_contrast_inputs():
    list_of_inputs = []
    
    # Input 1: Valid - 3D tensor with lower=0.1, upper=0.5, seed=1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.1
    upper = 0.5
    seed = 1
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Valid - 3D tensor with lower=0.2, upper=0.8, seed=2
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.8
    seed = 2
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Valid - 4D tensor with lower=0.0, upper=1.0, seed=3
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]], dtype=np.float32)
    lower = 0.0
    upper = 1.0
    seed = 3
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Valid - 3D tensor with negative values, lower=0.1, upper=0.7, seed=4
    image = np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    lower = 0.1
    upper = 0.7
    seed = 4
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Valid - 3D tensor with float values, lower=0.3, upper=0.9, seed=5
    image = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    lower = 0.3
    upper = 0.9
    seed = 5
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Valid - 3D tensor with values in range [0, 1], lower=0.2, upper=0.6, seed=6
    image = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    lower = 0.2
    upper = 0.6
    seed = 6
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Valid - 4D tensor with multiple channels, lower=0.1, upper=0.9, seed=7
    image = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]], dtype=np.float32)
    lower = 0.1
    upper = 0.9
    seed = 7
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Valid - 3D tensor with negative values, lower=0.4, upper=0.9, seed=8
    image = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    lower = 0.4
    upper = 0.9
    seed = 8
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Valid - 3D tensor with all zeros, lower=0.1, upper=0.6, seed=9
    image = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    lower = 0.1
    upper = 0.6
    seed = 9
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Valid - 3D tensor with mixed values, lower=0.2, upper=0.8, seed=10
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    lower = 0.2
    upper = 0.8
    seed = 10
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.random_contrast"] = tf_image_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_contrast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_contrast', generated_inputs['tf.image.random_contrast'], lib="tf", suffix=0)
