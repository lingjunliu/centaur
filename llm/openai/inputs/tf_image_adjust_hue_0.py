
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_image_adjust_hue_inputs():
    rs = np.random.RandomState(42)
    list_of_inputs = []

    # Input 1
    image = np.linspace(0.0, 1.0, num=12, dtype=np.float32).reshape(2, 2, 3)
    delta = 0.2
    name = "hue_pos_0_2_2x2x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 2
    image = np.arange(1, 19, dtype=np.int32).reshape(3, 2, 3)
    delta = -0.5
    name = "hue_neg_0_5_3x2x3_i32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 3
    image = rs.randint(0, 256, size=(1, 2, 2, 3), dtype=np.uint8)
    delta = 1.0
    name = "hue_pos_1_0_1x2x2x3_u8"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 4
    image = (rs.rand(4, 5, 3) * 10.0).astype(np.float64)
    delta = 0.75
    name = "hue_pos_0_75_4x5x3_f64"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 5
    image = rs.randint(-1000, 1000, size=(2, 3, 4, 4, 3), dtype=np.int32)
    delta = -1.0
    name = "hue_neg_1_0_2x3x4x4x3_i32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 6
    image = rs.rand(10, 3, 3).astype(np.float16)
    delta = 0.0
    name = "hue_zero_10x3x3_f16"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 7
    image = (np.arange(8 * 8 * 3, dtype=np.int64).reshape(8, 8, 3) + 1000)
    delta = 1.0 / 3.0
    name = "hue_pos_third_8x8x3_i64"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 8
    image = np.array([[[[100, 200, 30]]], [[[40, 50, 60]]]], dtype=np.uint8)
    delta = -0.75
    name = "hue_neg_0_75_2x1x1x3_u8"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 9
    image = rs.randint(0, 256, size=(5, 6, 3), dtype=np.uint8)
    delta = 0.999
    name = "hue_pos_0_999_5x6x3_u8"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 10
    image = (rs.rand(3, 3, 3) * 2.5).astype(np.float32)
    delta = -0.25
    name = "hue_neg_0_25_3x3x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 11
    image = (rs.rand(2, 4, 3) * 255).astype(np.float32)
    delta = 0.1
    name = "hue_pos_0_1_2x4x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 12
    image = rs.rand(1, 7, 7, 3).astype(np.float32)
    delta = -0.1
    name = "hue_neg_0_1_1x7x7x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.adjust_hue"] = tf_image_adjust_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_hue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_hue', generated_inputs['tf.image.adjust_hue'], lib="tf", suffix=0)
