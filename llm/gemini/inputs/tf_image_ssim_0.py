
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_ssim_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'img1': np.random.rand(1, 12, 12, 3).astype(np.float32),
        'img2': np.random.rand(1, 12, 12, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'img1': np.random.rand(2, 20, 20, 1).astype(np.float32) * 255.0,
        'img2': np.random.rand(2, 20, 20, 1).astype(np.float32) * 255.0,
        'max_val': 255.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'img1': np.random.rand(1, 15, 15, 3).astype(np.float32),
        'img2': np.random.rand(1, 15, 15, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 13,
        'filter_sigma': 2.0,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'img1': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'img2': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 7,
        'filter_sigma': 1.1,
        'k1': 0.02,
        'k2': 0.04,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'img1': np.random.rand(4, 16, 16, 4).astype(np.float32),
        'img2': np.random.rand(4, 16, 16, 4).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'img1': np.random.rand(1, 11, 11, 3).astype(np.float32),
        'img2': np.random.rand(1, 11, 11, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.05,
        'k2': 0.15,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'img1': np.random.rand(3, 32, 32, 3).astype(np.float32) * 100.0,
        'img2': np.random.rand(3, 32, 32, 3).astype(np.float32) * 100.0,
        'max_val': 100.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'img1': np.random.rand(2, 14, 14, 2).astype(np.float32),
        'img2': np.random.rand(2, 14, 14, 2).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 9,
        'filter_sigma': 1.2,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'img1': np.random.rand(1, 25, 25, 3).astype(np.float32),
        'img2': np.random.rand(1, 25, 25, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 21,
        'filter_sigma': 3.0,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'img1': np.random.rand(1, 11, 11, 1).astype(np.float32) * 2.0,
        'img2': np.random.rand(1, 11, 11, 1).astype(np.float32) * 2.0,
        'max_val': 2.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.005,
        'k2': 0.01,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.ssim"] = tf_image_ssim_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.ssim' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.ssim'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.ssim', generated_inputs['tf.image.ssim'], lib="tf", suffix=0)
