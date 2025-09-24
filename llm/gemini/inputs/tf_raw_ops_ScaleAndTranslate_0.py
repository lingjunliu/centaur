
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scale_and_translate_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 images, 4D input
    images = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    size = np.array([2, 2], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([0.0, 0.0], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: uint8 images, different size and translation, 4D input
    images = np.array([[[[5, 6], [7, 8]]]], dtype=np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    scale = np.array([1.5, 1.5], dtype=np.float32)
    translation = np.array([0.5, 0.5], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": False,
        "name": "scale_and_translate_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 images, 4D input
    images = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    translation = np.array([-0.5, -0.5], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 images, different kernel_type, 4D input
    images = np.array([[[[1, 2], [3, 4]]]], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    translation = np.array([1.0, 1.0], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More complex input with different shapes, 4D input
    images = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.int32)
    size = np.array([3, 4], dtype=np.int32)
    scale = np.array([0.75, 1.25], dtype=np.float32)
    translation = np.array([-0.2, 0.1], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Half type, 4D input
    images = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    translation = np.array([-0.5, -0.5], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bfloat16 type, 4D input
    images = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32).astype(np.float16) #Simulating bfloat16 with float16 as numpy doesnt support bfloat16
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    translation = np.array([-0.5, -0.5], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8 images, 4D input
    images = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)
    size = np.array([3, 3], dtype=np.int32)
    scale = np.array([1.5, 1.5], dtype=np.float32)
    translation = np.array([0.5, 0.5], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": False,
        "name": "scale_and_translate_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Remove uint16
    # Input 9: int16 images, 4D input
    images = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)
    size = np.array([2, 2], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([0.0, 0.0], dtype=np.float32)

    input_dict = {
        "images": images,
        "size": size,
        "scale": scale,
        "translation": translation,
        "kernel_type": "lanczos3",
        "antialias": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Remove uint16

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScaleAndTranslate"] = tf_raw_ops_scale_and_translate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScaleAndTranslate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScaleAndTranslate'.")

check_valid('tf.raw_ops.ScaleAndTranslate', generated_inputs['tf.raw_ops.ScaleAndTranslate'], lib="tf", suffix=0)
