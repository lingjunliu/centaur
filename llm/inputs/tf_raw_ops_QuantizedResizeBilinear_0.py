
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_quantizedresizebilinear_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedResizeBilinear function.
    """
    list_of_inputs = []

    # Input 1: Basic float32 upscaling (mimicking quint8)
    input_dict_1 = {
        'images': np.arange(48, dtype=np.float32).reshape((1, 4, 4, 3)),
        'size': np.array([8, 8], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(255.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': False,
        'name': "float32_upscale_quint8_mimic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float32 downscaling (mimicking qint32)
    input_dict_2 = {
        'images': np.arange(-50, 50, dtype=np.float32).reshape((1, 10, 10, 1)),
        'size': np.array([5, 5], dtype=np.int32),
        'min': np.array(-50.0, dtype=np.float32),
        'max': np.array(50.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': False,
        'name': "float32_downscale_qint32_mimic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float32 non-proportional resize
    input_dict_3 = {
        'images': np.random.rand(1, 6, 8, 3).astype(np.float32),
        'size': np.array([10, 5], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(1.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': False,
        'name': "float32_non_proportional"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: align_corners=True
    input_dict_4 = {
        'images': np.arange(16, dtype=np.float32).reshape((1, 4, 4, 1)),
        'size': np.array([7, 7], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(15.0, dtype=np.float32),
        'align_corners': True,
        'half_pixel_centers': False,
        'name': "align_corners_true"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: half_pixel_centers=True
    input_dict_5 = {
        'images': np.arange(16, dtype=np.float32).reshape((1, 4, 4, 1)),
        'size': np.array([7, 7], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(15.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': True,
        'name': "half_pixel_centers_true"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Batch size > 1
    input_dict_6 = {
        'images': np.zeros((2, 5, 5, 3), dtype=np.float32),
        'size': np.array([3, 3], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(255.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': False,
        'name': "batch_size_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 4 channels (e.g., RGBA)
    input_dict_7 = {
        'images': np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape((2, 3, 4, 4)),
        'size': np.array([6, 8], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(255.0, dtype=np.float32),
        'align_corners': True,
        'half_pixel_centers': False,
        'name': "rgba_channels"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Small 2x2 input image, with negative range
    input_dict_8 = {
        'images': np.array([[-10, 0], [10, 20]], dtype=np.float32).reshape((1, 2, 2, 1)),
        'size': np.array([4, 4], dtype=np.int32),
        'min': np.array(-50.0, dtype=np.float32),
        'max': np.array(50.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': True,
        'name': "small_image_neg_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large downscaling factor
    input_dict_9 = {
        'images': np.random.uniform(0, 255, size=(1, 128, 128, 1)).astype(np.float32),
        'size': np.array([8, 8], dtype=np.int32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(255.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': False,
        'name': "large_downscale"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Fully negative min/max range for float32
    input_dict_10 = {
        'images': np.arange(-20, -11, dtype=np.float32).reshape((1, 3, 3, 1)),
        'size': np.array([6, 6], dtype=np.int32),
        'min': np.array(-100.0, dtype=np.float32),
        'max': np.array(-10.0, dtype=np.float32),
        'align_corners': False,
        'half_pixel_centers': False,
        'name': "fully_negative_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: min and max are close together
    input_dict_11 = {
        'images': (np.random.rand(1, 7, 7, 3) * 0.2 + 0.4).astype(np.float32),
        'size': np.array([5, 5], dtype=np.int32),
        'min': np.array(0.4, dtype=np.float32),
        'max': np.array(0.6, dtype=np.float32),
        'align_corners': True,
        'half_pixel_centers': False,
        'name': "narrow_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedResizeBilinear"] = tf_raw_ops_quantizedresizebilinear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedResizeBilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedResizeBilinear'.")

check_valid('tf.raw_ops.QuantizedResizeBilinear', generated_inputs['tf.raw_ops.QuantizedResizeBilinear'], lib="tf", suffix=0)
