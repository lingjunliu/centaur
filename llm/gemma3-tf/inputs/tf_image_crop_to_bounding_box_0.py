
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_crop_to_bounding_box_inputs():
    list_of_inputs = []

    image = np.arange(1, 28, dtype=np.float32).reshape([3, 3, 3])
    offset_height = 0
    offset_width = 0
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

    image = np.random.rand(4, 5, 6, 3).astype(np.float32)
    offset_height = 1
    offset_width = 2
    target_height = 3
    target_width = 4
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(5, 5, 3).astype(np.float32)
    offset_height = 2
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

    image = np.random.rand(2, 4, 4, 1).astype(np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 4
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(6, 6, 3).astype(np.float32)
    offset_height = 3
    offset_width = 3
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

    image = np.random.rand(4, 4, 1).astype(np.float32)
    offset_height = 1
    offset_width = 1
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
    
    image = np.random.rand(5, 5, 3).astype(np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 5
    target_width = 5
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(3, 3, 3).astype(np.float32)
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

    image = np.random.rand(4, 5, 6, 2).astype(np.float32)
    offset_height = 2
    offset_width = 3
    target_height = 1
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 2, 2).astype(np.float32)
    offset_height = 0
    offset_width = 0
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
