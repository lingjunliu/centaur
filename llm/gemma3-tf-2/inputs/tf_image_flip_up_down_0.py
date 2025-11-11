
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_flip_up_down_inputs():
    list_of_inputs = []

    image1 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict = {'image': image1}
    list_of_inputs.append(input_dict)

    image2 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {'image': image2}
    list_of_inputs.append(input_dict)

    image3 = np.array([[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]], dtype=np.float32)
    input_dict = {'image': image3}
    list_of_inputs.append(input_dict)

    image4 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.int32)
    input_dict = {'image': image4}
    list_of_inputs.append(input_dict)

    image5 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    input_dict = {'image': image5}
    list_of_inputs.append(input_dict)

    image6 = np.random.rand(2, 3, 4, 1).astype(np.float32)
    input_dict = {'image': image6}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.image.flip_up_down"] = tf_image_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.flip_up_down'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.flip_up_down', generated_inputs['tf.image.flip_up_down'], lib="tf", suffix=0)
