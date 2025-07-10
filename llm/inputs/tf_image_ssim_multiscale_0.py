
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_ssim_multiscale_inputs():
    list_of_inputs = []

    # Input 1
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    img1 = np.random.rand(2, 128, 128, 1).astype(np.float32)
    img2 = np.random.rand(2, 128, 128, 1).astype(np.float32)
    max_val = 255.0
    power_factors = (0.1, 0.2, 0.3)
    filter_size = 7
    filter_sigma = 1.0
    k1 = 0.05
    k2 = 0.08
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    img1 = np.random.rand(4, 64, 64, 3).astype(np.float32)
    img2 = np.random.rand(4, 64, 64, 3).astype(np.float32)
    max_val = 100.0
    power_factors = (0.2, 0.3, 0.5)
    filter_size = 5
    filter_sigma = 0.8
    k1 = 0.005
    k2 = 0.015
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    img1 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    max_val = 50.0
    power_factors = (0.5, 0.5)
    filter_size = 3
    filter_sigma = 0.5
    k1 = 0.001
    k2 = 0.003
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    img1 = np.random.rand(8, 128, 128, 3).astype(np.float32)
    img2 = np.random.rand(8, 128, 128, 3).astype(np.float32)
    max_val = 255.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    img1 = np.random.rand(1, 64, 64, 1).astype(np.float32)
    img2 = np.random.rand(1, 64, 64, 1).astype(np.float32)
    max_val = 1.0
    power_factors = (0.1, 0.9)
    filter_size = 5
    filter_sigma = 1.0
    k1 = 0.001
    k2 = 0.003
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    img1 = np.random.rand(2, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(2, 32, 32, 3).astype(np.float32)
    max_val = 200.0
    power_factors = (0.2, 0.3, 0.5)
    filter_size = 3
    filter_sigma = 0.8
    k1 = 0.02
    k2 = 0.06
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    img1 = np.random.rand(4, 256, 256, 1).astype(np.float32)
    img2 = np.random.rand(4, 256, 256, 1).astype(np.float32)
    max_val = 150.0
    power_factors = (0.1, 0.2, 0.3, 0.4)
    filter_size = 7
    filter_sigma = 1.2
    k1 = 0.03
    k2 = 0.09
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    img1 = np.random.rand(16, 64, 64, 3).astype(np.float32)
    img2 = np.random.rand(16, 64, 64, 3).astype(np.float32)
    max_val = 255.0
    power_factors = (0.05, 0.25, 0.35, 0.20, 0.15)
    filter_size = 9
    filter_sigma = 1.3
    k1 = 0.015
    k2 = 0.045
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    img1 = np.random.rand(1, 28, 28, 1).astype(np.float32)
    img2 = np.random.rand(1, 28, 28, 1).astype(np.float32)
    max_val = 100.0
    power_factors = (0.33, 0.33, 0.34)
    filter_size = 5
    filter_sigma = 0.9
    k1 = 0.007
    k2 = 0.021
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 11: Ensure image dimensions are at least filter_size + 6
    img1 = np.random.rand(1, 17, 17, 1).astype(np.float32)
    img2 = np.random.rand(1, 17, 17, 1).astype(np.float32)
    max_val = 100.0
    power_factors = (0.33, 0.33, 0.34)
    filter_size = 11
    filter_sigma = 0.9
    k1 = 0.007
    k2 = 0.021
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    max_val = 255.0
    power_factors = (0.0448, 0.2856)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    input_dict = {"img1": img1, "img2": img2, "max_val": max_val, "power_factors": power_factors, "filter_size": filter_size, "filter_sigma": filter_sigma, "k1": k1, "k2": k2}
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
