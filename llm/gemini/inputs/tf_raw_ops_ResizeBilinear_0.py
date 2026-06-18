
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeBilinear_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "images": np.ones((1, 2, 2, 1), dtype=np.float32),
        "size": np.array([4, 4], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": False,
        "name": "resize_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "images": np.zeros((2, 3, 3, 3), dtype=np.uint8),
        "size": np.array([6, 6], dtype=np.int32),
        "align_corners": True,
        "half_pixel_centers": False,
        "name": "resize_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "images": np.arange(48, dtype=np.uint8).reshape((1, 4, 4, 3)),
        "size": np.array([2, 2], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "images": np.ones((2, 5, 5, 2), dtype=np.float32) * -1.5,
        "size": np.array([10, 10], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "images": np.ones((1, 1, 1, 1), dtype=np.int32),
        "size": np.array([3, 3], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": False,
        "name": "resize_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "images": np.arange(16, dtype=np.float32).reshape((1, 2, 2, 4)),
        "size": np.array([5, 5], dtype=np.int32),
        "align_corners": True,
        "half_pixel_centers": False,
        "name": "resize_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "images": np.zeros((1, 8, 8, 1), dtype=np.int32),
        "size": np.array([4, 4], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "images": np.ones((2, 2, 2, 3), dtype=np.float32) * 3.14,
        "size": np.array([4, 4], dtype=np.int32),
        "align_corners": True,
        "half_pixel_centers": False,
        "name": "resize_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "images": np.ones((1, 3, 3, 2), dtype=np.uint8),
        "size": np.array([1, 1], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": False,
        "name": "resize_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "images": np.arange(32, dtype=np.float32).reshape((1, 4, 4, 2)),
        "size": np.array([8, 8], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ResizeBilinear"] = tf_raw_ops_ResizeBilinear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ResizeBilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ResizeBilinear'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ResizeBilinear', generated_inputs['tf.raw_ops.ResizeBilinear'], lib="tf", suffix=0)
