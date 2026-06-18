
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_image_resize_inputs():
    list_of_inputs = []

    # Input 1: Standard 4-D float32 batch, bilinear, no antialias, size larger (upsampling)
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    size = np.array([20, 20], dtype=np.int32)
    method = "bilinear"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_1"
    
    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 2: 3-D uint8 image, bicubic, preserve aspect ratio, size smaller
    images = np.random.randint(0, 256, size=(15, 30, 3)).astype(np.uint8)
    size = np.array([10, 10], dtype=np.int32)
    method = "bicubic"
    preserve_aspect_ratio = True
    antialias = False
    name = "resize_2"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 3: 4-D int32 batch, nearest neighbor, antialias has no effect
    images = np.random.randint(-100, 100, size=(1, 5, 5, 1)).astype(np.int32)
    size = np.array([8, 8], dtype=np.int32)
    method = "nearest"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_3"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 4: 3-D float64 image, lanczos3, antialias=True for downsampling
    images = np.random.rand(32, 32, 4).astype(np.float64)
    size = np.array([16, 16], dtype=np.int32)
    method = "lanczos3"
    preserve_aspect_ratio = False
    antialias = True
    name = "resize_4"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 5: 4-D float16, area method, preserve aspect ratio
    images = np.random.rand(4, 24, 16, 3).astype(np.float16)
    size = np.array([12, 12], dtype=np.int32)
    method = "area"
    preserve_aspect_ratio = True
    antialias = False
    name = "resize_5"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 6: 3-D float32 single channel, gaussian, preserve aspect ratio
    images = np.random.rand(100, 50, 1).astype(np.float32)
    size = np.array([40, 40], dtype=np.int32)
    method = "gaussian"
    preserve_aspect_ratio = True
    antialias = True
    name = "resize_6"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 7: 4-D float32, mitchellcubic, downsampling
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    size = np.array([32, 32], dtype=np.int32)
    method = "mitchellcubic"
    preserve_aspect_ratio = False
    antialias = True
    name = "resize_7"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 8: 3-D uint8, lanczos5, no aspect ratio preservation
    images = np.random.randint(0, 256, size=(40, 40, 3)).astype(np.uint8)
    size = np.array([80, 80], dtype=np.int32)
    method = "lanczos5"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_8"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 9: 4-D float32, bilinear with antialias and aspect ratio preserved
    images = np.random.rand(1, 128, 64, 3).astype(np.float32)
    size = np.array([32, 32], dtype=np.int32)
    method = "bilinear"
    preserve_aspect_ratio = True
    antialias = True
    name = "resize_9"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 10: 3-D int32, nearest neighbor, upsampling
    images = np.random.randint(-10, 10, size=(4, 4, 2)).astype(np.int32)
    size = np.array([12, 12], dtype=np.int32)
    method = "nearest"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_10"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.image.resize"] = tf_image_resize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.resize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.resize', generated_inputs['tf.image.resize'], lib="tf", suffix=0)
