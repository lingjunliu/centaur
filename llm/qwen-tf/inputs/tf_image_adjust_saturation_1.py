
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_image_adjust_saturation_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    saturation_factor = np.float32(0.0)
    name = "zero_sat_uint8_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(3, 5, 3).astype(np.float32)
    saturation_factor = np.float32(0.5)
    name = "half_sat_float32_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.randint(0, 256, size=(2, 3, 4, 3), dtype=np.uint8)
    saturation_factor = np.float64(2.0)
    name = "double_sat_uint8_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(1, 2, 2, 3).astype(np.float32)
    saturation_factor = np.float32(1.0)
    name = "no_change_float32_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(5, 2, 3).astype(np.float16)
    saturation_factor = np.float16(3.5)
    name = "high_sat_float16_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(3, 4, 5, 3).astype(np.float64)
    saturation_factor = np.float64(10.0)
    name = "very_high_sat_float64_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.tile(np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3, 1), (1, 1, 3))
    saturation_factor = np.float32(1.25)
    name = "grayscale_like_float32_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.randint(0, 256, size=(2, 1, 1, 3), dtype=np.uint8)
    saturation_factor = np.float32(4.0)
    name = "tiny_spatial_uint8_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(64, 64, 3).astype(np.float32)
    saturation_factor = np.float32(1.25)
    name = "mid_sat_float32_3d_large"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.randint(0, 256, size=(5, 8, 8, 3), dtype=np.uint8)
    saturation_factor = np.float64(0.25)
    name = "quarter_sat_uint8_4d_batch5"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = (np.random.rand(10, 10, 3).astype(np.float32) * 2.0)
    saturation_factor = np.float32(2.5)
    name = "over_one_range_float32_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.adjust_saturation_1"] = tf_image_adjust_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_saturation_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_saturation_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_saturation', generated_inputs['tf.image.adjust_saturation_1'], lib="tf", suffix=1)
