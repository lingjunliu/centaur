
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def _create_jpeg(height, width, channels=3):
    """Helper to create a JPEG image using TensorFlow and return its byte content."""
    image_array = np.zeros((height, width, channels), dtype=np.uint8)
    jpeg_bytes_tensor = tf.image.encode_jpeg(image_array)
    return jpeg_bytes_tensor.numpy()

def get_tf_io_extract_jpeg_shape_inputs():
    """
    Generates a list of valid inputs for the tf.io.extract_jpeg_shape function.
    """
    # Create some JPEG byte strings to use in the inputs.
    jpeg_1x1_rgb = _create_jpeg(1, 1, channels=3)
    jpeg_20x10_l = _create_jpeg(20, 10, channels=1)  # Grayscale
    jpeg_50x100_rgb = _create_jpeg(50, 100, channels=3)
    jpeg_256x256_rgb = _create_jpeg(256, 256, channels=3)
    jpeg_64x32_rgb = _create_jpeg(64, 32, channels=3)

    list_of_inputs = []

    # Input 1: Basic case with a 1x1 RGB image, default output_type and name
    input_dict = {
        'contents': np.array(jpeg_1x1_rgb),
        'output_type': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same image, but with output_type=int64
    input_dict = {
        'contents': np.array(jpeg_1x1_rgb),
        'output_type': np.int64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Same image, with a specific name
    input_dict = {
        'contents': np.array(jpeg_1x1_rgb),
        'output_type': np.int32,
        'name': 'extract_shape_1x1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale image (20x10), default output_type
    input_dict = {
        'contents': np.array(jpeg_20x10_l),
        'output_type': np.int32,
        'name': 'grayscale_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Grayscale image, output_type=int64
    input_dict = {
        'contents': np.array(jpeg_20x10_l),
        'output_type': np.int64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Medium RGB image (50x100), default output_type, empty name
    input_dict = {
        'contents': np.array(jpeg_50x100_rgb),
        'output_type': np.int32,
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Medium RGB image, output_type=int64, with a name
    input_dict = {
        'contents': np.array(jpeg_50x100_rgb),
        'output_type': np.int64,
        'name': 'medium_image_shape_64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger RGB image (256x256), default output_type
    input_dict = {
        'contents': np.array(jpeg_256x256_rgb),
        'output_type': np.int32,
        'name': 'large_image'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger RGB image, output_type=int64
    input_dict = {
        'contents': np.array(jpeg_256x256_rgb),
        'output_type': np.int64,
        'name': 'large_image_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another RGB image (64x32), output_type=np.int32
    input_dict = {
        'contents': np.array(jpeg_64x32_rgb),
        'output_type': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Another RGB image (64x32), output_type=np.int64
    input_dict = {
        'contents': np.array(jpeg_64x32_rgb),
        'output_type': np.int64,
        'name': 'another_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.extract_jpeg_shape"] = get_tf_io_extract_jpeg_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.extract_jpeg_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.extract_jpeg_shape'.")

check_valid('tf.io.extract_jpeg_shape', generated_inputs['tf.io.extract_jpeg_shape'], lib="tf", suffix=0)
