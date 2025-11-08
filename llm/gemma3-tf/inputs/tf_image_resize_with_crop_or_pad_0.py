
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []

    image1 = np.arange(75).reshape(5, 5, 3).astype(np.int64)
    target_height1 = 3
    target_width1 = 3
    input_dict1 = {"image": image1, "target_height": target_height1, "target_width": target_width1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.arange(1, 28).reshape(3, 3, 3).astype(np.int64)
    target_height2 = 5
    target_width2 = 5
    input_dict2 = {"image": image2, "target_height": target_height2, "target_width": target_width2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(2, 2, 2, 3).astype(np.float32)
    target_height3 = 4
    target_width3 = 4
    input_dict3 = {"image": image3, "target_height": target_height3, "target_width": target_width3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.arange(12).reshape(2, 3, 2).astype(np.int32)
    target_height4 = 1
    target_width4 = 1
    input_dict4 = {"image": image4, "target_height": target_height4, "target_width": target_width4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.zeros((4, 4, 1)).astype(np.float64)
    target_height5 = 8
    target_width5 = 8
    input_dict5 = {"image": image5, "target_height": target_height5, "target_width": target_width5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.randint(0, 256, size=(6, 6, 3), dtype=np.uint8)
    target_height6 = 3
    target_width6 = 3
    input_dict6 = {"image": image6, "target_height": target_height6, "target_width": target_width6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.arange(16).reshape(4, 4, 1).astype(np.int64)
    target_height7 = 2
    target_width7 = 2
    input_dict7 = {"image": image7, "target_height": target_height7, "target_width": target_width7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["tf.image.resize_with_crop_or_pad"] = tf_image_resize_with_crop_or_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.resize_with_crop_or_pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize_with_crop_or_pad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.resize_with_crop_or_pad', generated_inputs['tf.image.resize_with_crop_or_pad'], lib="tf", suffix=0)
