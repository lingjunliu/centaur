
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_image_crop_to_bounding_box_inputs():
    list_of_inputs = []

    # Input 1
    image = np.arange(5*7*3, dtype=np.float32).reshape(5, 7, 3)
    offset_height = np.int32(0)
    offset_width = np.int32(0)
    target_height = np.int32(2)
    target_width = np.int32(4)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = (np.random.rand(10, 10, 1) * 255).astype(np.uint8)
    offset_height = 3
    offset_width = 4
    target_height = 5
    target_width = 6
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.arange(4*5*2, dtype=np.int32).reshape(4, 5, 2)
    offset_height = np.int64(1)
    offset_width = np.int64(1)
    target_height = np.int64(3)
    target_width = np.int64(4)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = (np.random.randn(2, 8, 9, 3)).astype(np.float16)
    offset_height = 2
    offset_width = 3
    target_height = 4
    target_width = 5
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = (np.random.randint(0, 256, size=(1, 6, 6, 4))).astype(np.uint8)
    offset_height = 0
    offset_width = 0
    target_height = 6
    target_width = 6
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = (np.random.rand(3, 3, 2) > 0.5).astype(np.float32)
    offset_height = np.int32(1)
    offset_width = np.int32(1)
    target_height = np.int32(2)
    target_width = np.int32(2)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.random.randint(-1000, 1000, size=(3, 7, 8, 1), dtype=np.int64)
    offset_height = 5
    offset_width = 6
    target_height = 2
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.linspace(0, 1, 100*50*3, dtype=np.float64).reshape(100, 50, 3)
    offset_height = np.int64(10)
    offset_width = np.int64(5)
    target_height = np.int64(80)
    target_width = np.int64(40)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.arange(12*12*1, dtype=np.float32).reshape(12, 12, 1)
    offset_height = 11
    offset_width = 11
    target_height = 1
    target_width = 1
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    image = np.random.randint(-200, 200, size=(5, 20, 30, 3), dtype=np.int32)
    offset_height = 0
    offset_width = 10
    target_height = 10
    target_width = 20
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    image = np.arange(2*5*4, dtype=np.int32).reshape(2, 5, 4)
    offset_height = np.int32(0)
    offset_width = np.int32(1)
    target_height = np.int32(2)
    target_width = np.int32(3)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    image = np.random.randn(2, 2, 2, 2).astype(np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 1
    target_width = 1
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.crop_to_bounding_box"] = tf_image_crop_to_bounding_box_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.crop_to_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.crop_to_bounding_box'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.crop_to_bounding_box', generated_inputs['tf.image.crop_to_bounding_box'], lib="tf", suffix=0)
