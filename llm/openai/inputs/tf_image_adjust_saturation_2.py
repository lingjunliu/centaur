
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_saturation_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[0, 128, 255],
                       [30, 60, 90]],
                      [[200, 150, 100],
                       [255, 0, 50]]], dtype=np.uint8)
    saturation_factor = np.array(0.0, dtype=np.float32)
    name = "case1_zero_sat"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 2
    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    saturation_factor = np.array(0.5, dtype=np.float32)
    name = "case2_half"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 3
    image = np.linspace(-1.0, 1.0, num=27, dtype=np.float32).reshape(3, 3, 3)
    saturation_factor = np.array(1.0, dtype=np.float32)
    name = "case3_negative_values"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 4
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    saturation_factor = np.array(2.0, dtype=np.float32)
    name = "case4_float32_small_image"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 5
    rng = np.random.RandomState(0)
    image = rng.rand(2, 3, 4, 3).astype(np.float32)
    saturation_factor = np.array(1.5, dtype=np.float32)
    name = "case5_batched_float32"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 6
    rng = np.random.RandomState(1)
    image = rng.randint(0, 256, size=(3, 2, 2, 3), dtype=np.uint8)
    saturation_factor = np.array(1.2, dtype=np.float32)
    name = "case6_batched_uint8"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 7
    rng = np.random.RandomState(2)
    image = (rng.rand(6, 5, 3) * 255).astype(np.float32)
    saturation_factor = np.array(3.0, dtype=np.float32)
    name = "case7_float32_high_saturation"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 8
    image = np.arange(27, dtype=np.uint8).reshape(1, 3, 3, 3)
    saturation_factor = np.array(0.75, dtype=np.float32)
    name = "case8_single_batch_uint8"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 9
    image = np.zeros((8, 8, 3), dtype=np.float32)
    saturation_factor = np.array(5.0, dtype=np.float32)
    name = "case9_zero_image_high_factor"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 10
    rng = np.random.RandomState(3)
    image = rng.rand(4, 4, 3).astype(np.float32)
    saturation_factor = np.array(10.0, dtype=np.float32)
    name = "case10_extreme_factor"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 11
    grad = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    r = np.tile(grad[:, None], (1, 10))
    g = np.tile(grad[None, :], (10, 1))
    b = np.flipud(r)
    image = np.stack([r, g, b], axis=-1).astype(np.float32)
    saturation_factor = np.array(2.5, dtype=np.float32)
    name = "case11_gradient_float32"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 12
    image = np.array([[[1000.0, 200.0, 50.0],
                       [500.0, 500.0, 500.0]],
                      [[-100.0, 0.0, 100.0],
                       [1e6, 1e6 - 1e3, 1e6 - 2e3]]], dtype=np.float32)
    saturation_factor = np.array(0.1, dtype=np.float32)
    name = "case12_large_values_float32"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.adjust_saturation_2"] = tf_image_adjust_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_saturation_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_saturation_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_saturation', generated_inputs['tf.image.adjust_saturation_2'], lib="tf", suffix=2)
