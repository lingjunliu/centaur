
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

def tf_image_random_contrast_inputs():
    list_of_inputs = []

    image = np.linspace(0, 1, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    lower = np.float32(0.5)
    upper = np.float32(1.5)
    seed = np.int32(42)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.arange(10 * 10, dtype=np.uint8).reshape(10, 10, 1)
    lower = np.float32(0.0)
    upper = np.float32(2.0)
    seed = np.int64(123)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(-5, 5, 2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    lower = np.float32(0.2)
    upper = np.float32(0.8)
    seed = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 2, 3 * 2 * 16 * 16 * 3, dtype=np.float32).reshape(3, 2, 16, 16, 3)
    lower = np.float32(0.1)
    upper = np.float32(1.0)
    seed = np.int64(999999)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 255, 7 * 7 * 4, dtype=np.float32).reshape(7, 7, 4)
    lower = np.float32(1.2)
    upper = np.float32(1.3)
    seed = np.int32(31415)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.arange(5 * 32 * 32 * 3, dtype=np.uint8) % 256).reshape(5, 32, 32, 3)
    lower = np.float32(0.01)
    upper = np.float32(0.99)
    seed = np.int64(7)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(-1, 1, 2 * 3 * 4 * 10 * 10 * 3, dtype=np.float32).reshape(2, 3, 4, 10, 10, 3)
    lower = np.float32(0.75)
    upper = np.float32(1.25)
    seed = np.int64(2024)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    lower = np.float32(0.0001)
    upper = np.float32(0.0002)
    seed = np.int32(555)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 10, 4 * 6 * 6 * 2, dtype=np.float32).reshape(4, 6, 6, 2)
    lower = np.float32(2.0)
    upper = np.float32(3.0)
    seed = np.int64(88)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 1, 1 * 10 * 10 * 5, dtype=np.float32).reshape(1, 10, 10, 5)
    lower = np.float32(0.3)
    upper = np.float32(0.3001)
    seed = np.int32(999)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 100, 12 * 9 * 1, dtype=np.float32).reshape(12, 9, 1)
    lower = np.float32(0.4)
    upper = np.float32(1.8)
    seed = np.int64(321)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(-10, 10, 3 * 4 * 4 * 3, dtype=np.float32).reshape(3, 4, 4, 3)
    lower = np.float32(0.6)
    upper = np.float32(1.4)
    seed = np.int32(1024)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.random_contrast"] = tf_image_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_contrast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_contrast', generated_inputs['tf.image.random_contrast'], lib="tf", suffix=0)
