
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []

    image = np.arange(75, dtype=np.float32).reshape(5, 5, 3)
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(1, 28, dtype=np.int32).reshape(3, 3, 3)
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.linspace(-1.0, 1.0, 7 * 4 * 1, dtype=np.float32).reshape(7, 4, 1)
    target_height = np.int64(5)
    target_width = np.int64(6)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.randn(2, 6, 9, 3).astype(np.float32)
    target_height = np.int32(8)
    target_width = np.int32(5)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(1 * 3 * 5 * 2, dtype=np.int32).reshape(1, 3, 5, 2)
    target_height = 2
    target_width = 2
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.ones((4, 4, 1), dtype=np.float32)
    target_height = 6
    target_width = 7
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.zeros((3, 2, 2, 3), dtype=np.float64)
    target_height = np.int64(3)
    target_width = np.int64(5)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.uniform(-100, 100, size=(9, 7, 5)).astype(np.float32)
    target_height = 9
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(-60, 15, dtype=np.int64).reshape(5, 5, 3)
    target_height = 7
    target_width = 6
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.normal(0, 1, size=(10, 8, 4)).astype(np.float32)
    target_height = 10
    target_width = 8
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.zeros((4, 5, 10, 3), dtype=np.int32)
    target_height = np.int32(5)
    target_width = np.int32(12)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.randint(-50, 50, size=(2, 11, 3, 1)).astype(np.int32)
    target_height = 5
    target_width = 6
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
