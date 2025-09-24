
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_convert_image_dtype_inputs():
    list_of_inputs = []

    # Input 1: uint8 to float32
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    dtype = tf.float32
    saturate = False
    name = "uint8_to_float32"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 to int8
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.0, 0.2]]], dtype=np.float32)
    dtype = tf.int8
    saturate = True
    name = "float32_to_int8"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int16 to float16
    image = np.array([[[1000, 2000], [3000, 4000]], [[5000, 6000], [7000, 8000]]], dtype=np.int16)
    dtype = tf.float16
    saturate = False
    name = "int16_to_float16"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 to uint8 with saturation
    image = np.array([[[0.1, 0.9], [0.3, 0.7]], [[0.5, 0.5], [0.2, 0.8]]], dtype=np.float64)
    dtype = tf.uint8
    saturate = True
    name = "float64_to_uint8_sat"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32 to float32
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dtype = tf.float32
    saturate = False
    name = "int32_to_float32"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 to float64
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dtype = tf.float64
    saturate = False
    name = "int64_to_float64"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 to int32, with saturation
    image = np.array([[[0.1, 0.2], [0.9, 1.0]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    dtype = tf.int32
    saturate = True
    name = "float32_to_int32_sat"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 to int64
    image = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    dtype = tf.int64
    saturate = False
    name = "float32_to_int64"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: float16 to uint8 with saturation
    image = np.array([[[0.1, 0.9], [0.3, 0.7]], [[0.5, 0.5], [0.2, 0.8]]], dtype=np.float16)
    dtype = tf.uint8
    saturate = True
    name = "float16_to_uint8_sat"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int8 to uint16 - Adding a valid conversion within the documented supported types
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int8)
    dtype = tf.uint16
    saturate = True
    name = "int8_to_uint16"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: float16 to float32
    image = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    dtype = tf.float32
    saturate = False
    name = "float16_to_float32"
    input_dict = {"image": image, "dtype": dtype, "saturate": saturate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.convert_image_dtype"] = tf_image_convert_image_dtype_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.convert_image_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.convert_image_dtype'.")

check_valid('tf.image.convert_image_dtype', generated_inputs['tf.image.convert_image_dtype'], lib="tf", suffix=0)
