
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_psnr_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.rand(1, 32, 32, 3).astype(np.float32)
    b = np.random.rand(1, 32, 32, 3).astype(np.float32)
    max_val = 1.0
    name = "psnr_1"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.random.rand(2, 64, 64, 1).astype(np.float32)
    b = np.random.rand(2, 64, 64, 1).astype(np.float32)
    max_val = 255.0
    name = "psnr_2"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.rand(4, 128, 128, 3).astype(np.float32)
    b = np.random.rand(4, 128, 128, 3).astype(np.float32)
    max_val = 10.0
    name = "psnr_3"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.rand(1, 256, 256, 1).astype(np.float32)
    b = np.random.rand(1, 256, 256, 1).astype(np.float32)
    max_val = 1.0
    name = None
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.random.rand(8, 32, 32, 3).astype(np.float32)
    b = np.random.rand(8, 32, 32, 3).astype(np.float32)
    max_val = 255.0
    name = "psnr_5"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.random.rand(1, 16, 16, 1).astype(np.float32)
    b = np.random.rand(1, 16, 16, 1).astype(np.float32)
    max_val = 1.0
    name = "psnr_6"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.random.rand(2, 8, 8, 3).astype(np.float32)
    b = np.random.rand(2, 8, 8, 3).astype(np.float32)
    max_val = 255.0
    name = "psnr_7"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.random.rand(4, 4, 4, 1).astype(np.float32)
    b = np.random.rand(4, 4, 4, 1).astype(np.float32)
    max_val = 1.0
    name = "psnr_8"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.rand(1, 32, 32, 3).astype(np.float32) * 255.0
    b = np.random.rand(1, 32, 32, 3).astype(np.float32) * 255.0
    max_val = 255.0
    name = "psnr_9"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    a = np.random.rand(1, 32, 32, 3).astype(np.float32) * 1.0
    b = np.random.rand(1, 32, 32, 3).astype(np.float32) * 1.0
    max_val = 1.0
    name = "psnr_10"
    input_dict = {"a": a, "b": b, "max_val": max_val, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.psnr"] = tf_image_psnr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.psnr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.psnr'.")

check_valid('tf.image.psnr', generated_inputs['tf.image.psnr'], lib="tf", suffix=0)
