
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_ssim_multiscale_inputs():
    list_of_inputs = []

    # Input 1
    img1 = np.abs(np.random.rand(1, 256, 256, 3).astype(np.float32))
    img2 = np.abs(np.random.rand(1, 256, 256, 3).astype(np.float32))
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    img1 = np.abs(np.random.rand(2, 128, 128, 1).astype(np.float32))
    img2 = np.abs(np.random.rand(2, 128, 128, 1).astype(np.float32))
    max_val = 255.0
    power_factors = (0.1, 0.2, 0.3, 0.2, 0.2)
    filter_size = 7
    filter_sigma = 1.0
    k1 = 0.005
    k2 = 0.02

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    img1 = np.abs(np.random.rand(4, 64, 64, 3).astype(np.float32))
    img2 = np.abs(np.random.rand(4, 64, 64, 3).astype(np.float32))
    max_val = 65535.0
    power_factors = (0.2, 0.2, 0.2, 0.2, 0.2)
    filter_size = 5
    filter_sigma = 0.8
    k1 = 0.02
    k2 = 0.06

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    img1 = np.abs(np.random.rand(1, 32, 32, 1).astype(np.float32))
    img2 = np.abs(np.random.rand(1, 32, 32, 1).astype(np.float32))
    max_val = 1023.0
    power_factors = (0.3, 0.3, 0.2, 0.1, 0.1)
    filter_size = 9
    filter_sigma = 1.2
    k1 = 0.001
    k2 = 0.003

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    img1 = np.abs(np.random.rand(8, 16, 16, 3).astype(np.float32))
    img2 = np.abs(np.random.rand(8, 16, 16, 3).astype(np.float32))
    max_val = 2047.0
    power_factors = (0.4, 0.3, 0.15, 0.1, 0.05)
    filter_size = 3
    filter_sigma = 0.5
    k1 = 0.03
    k2 = 0.09

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.ssim_multiscale"] = tf_image_ssim_multiscale_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.ssim_multiscale' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.ssim_multiscale'.")

check_valid('tf.image.ssim_multiscale', generated_inputs['tf.image.ssim_multiscale'], lib="tf", suffix=0)
