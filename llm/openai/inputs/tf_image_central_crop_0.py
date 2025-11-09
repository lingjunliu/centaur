
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_image_central_crop_inputs():
    list_of_inputs = []

    image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    central_fraction = np.float32(0.5)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.randn(5, 7, 1).astype(np.float64)
    central_fraction = np.float64(0.8)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(2, 8, 6, 3).astype(np.float32)
    central_fraction = np.float32(1.0)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.linspace(-1, 1, num=9 * 9 * 4, dtype=np.float16).reshape(9, 9, 4)
    central_fraction = np.float16(0.33)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.arange(3 * 10 * 10 * 1, dtype=np.float32).reshape(3, 10, 10, 1)
    central_fraction = np.float32(0.25)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.uniform(-5, 5, size=(7, 3, 2)).astype(np.float32)
    central_fraction = np.float32(0.95)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.randn(1, 12, 5, 3).astype(np.float64)
    central_fraction = np.float64(0.6)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(100, 100, 3).astype(np.float32)
    central_fraction = np.float32(0.01)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(5, 13, 13, 3).astype(np.float32)
    central_fraction = np.float32(2.0 / 3.0)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(2, 50, 1).astype(np.float32)
    central_fraction = np.float32(0.5)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = (np.random.rand(4, 15, 20, 2).astype(np.float16) * 2 - 1).astype(np.float16)
    central_fraction = np.float16(0.75)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(11, 11, 3).astype(np.float64)
    central_fraction = np.float64(0.999999)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    return list_of_inputs

generated_inputs["tf.image.central_crop"] = tf_image_central_crop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.central_crop' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.central_crop'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.central_crop', generated_inputs['tf.image.central_crop'], lib="tf", suffix=0)
