
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScaleAndTranslate_inputs():
    list_of_inputs = []

    # Input 1
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([1.0, 1.0], dtype=np.float32)
    kernel_type = "lanczos3"
    antialias = True
    name = "scale_and_translate_1"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    images = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    size = np.array([2, 2], dtype=np.int32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    translation = np.array([0.0, 0.0], dtype=np.float32)
    kernel_type = "bicubic"
    antialias = False
    name = "scale_and_translate_2"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    images = np.array([1, 2, 3, 4], dtype=np.float32)
    size = np.array([8], dtype=np.int32)
    scale = np.array([4.0], dtype=np.float32)
    translation = np.array([2.0], dtype=np.float32)
    kernel_type = "gaussian"
    antialias = True
    name = "scale_and_translate_3"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([1.0, 1.0], dtype=np.float32)
    kernel_type = "lanczos5"
    antialias = True
    name = "scale_and_translate_4"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.int32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    images = np.array([[[1, 2], [3, 4]]], dtype=np.float64)
    size = np.array([2, 2], dtype=np.int32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    translation = np.array([0.0, 0.0], dtype=np.float32)
    kernel_type = "mitchellcubic"
    antialias = False
    name = "scale_and_translate_5"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float64),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    images = np.array([1, 2, 3, 4], dtype=np.int8)
    size = np.array([8], dtype=np.int32)
    scale = np.array([4.0], dtype=np.float32)
    translation = np.array([2.0], dtype=np.float32)
    kernel_type = "nearest"
    antialias = True
    name = "scale_and_translate_6"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.int8),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    images = np.random.rand(1, 10, 10).astype(np.float32)
    size = np.array([20, 20], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([0.0, 0.0], dtype=np.float32)
    kernel_type = "lanczos3"
    antialias = True
    name = "scale_and_translate_7"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    images = np.random.rand(10, 10).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    translation = np.array([2.0, 2.0], dtype=np.float32)
    kernel_type = "lanczos5"
    antialias = False
    name = "scale_and_translate_8"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    images = np.random.rand(1, 10).astype(np.float32)
    size = np.array([5], dtype=np.int32)
    scale = np.array([0.5], dtype=np.float32)
    translation = np.array([2.0], dtype=np.float32)
    kernel_type = "gaussian"
    antialias = False
    name = "scale_and_translate_9"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    images = np.random.rand(10,).astype(np.float32)
    size = np.array([5], dtype=np.int32)
    scale = np.array([0.5], dtype=np.float32)
    translation = np.array([2.0], dtype=np.float32)
    kernel_type = "keyscubic"
    antialias = False
    name = "scale_and_translate_10"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.float32),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Test with uint8 images
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([1.0, 1.0], dtype=np.float32)
    kernel_type = "lanczos3"
    antialias = True
    name = "scale_and_translate_11"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.uint8),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Test with int64 images
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([1.0, 1.0], dtype=np.float32)
    kernel_type = "lanczos3"
    antialias = True
    name = "scale_and_translate_12"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.int64),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Test with bfloat16
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.bfloat16)
    size = np.array([4, 4], dtype=np.int32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    translation = np.array([1.0, 1.0], dtype=np.float32)
    kernel_type = "lanczos3"
    antialias = True
    name = "scale_and_translate_13"

    input_dict = {
        "images": tf.convert_to_tensor(images, dtype=tf.bfloat16),
        "size": tf.convert_to_tensor(size, dtype=tf.int32),
        "scale": tf.convert_to_tensor(scale, dtype=tf.float32),
        "translation": tf.convert_to_tensor(translation, dtype=tf.float32),
        "kernel_type": kernel_type,
        "antialias": antialias,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScaleAndTranslate"] = tf_raw_ops_ScaleAndTranslate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScaleAndTranslate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScaleAndTranslate'.")

check_valid('tf.raw_ops.ScaleAndTranslate', generated_inputs['tf.raw_ops.ScaleAndTranslate'], lib="tf", suffix=0)
