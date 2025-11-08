
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_DrawBoundingBoxes_inputs():
    list_of_inputs = []

    images = np.array([[[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]],
                        [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]]], dtype=np.float32)
    boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    input_dict = {
        "name": "case1_basic_rgb",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.linspace(0, 1, 2 * 5 * 4 * 1, dtype=np.float32).reshape(2, 5, 4, 1)
    boxes = np.array([
        [[0.0, 0.0, 1.0, 1.0], [0.2, 0.2, 0.6, 0.8]],
        [[0.1, 0.1, 0.9, 0.9], [0.0, 0.5, 1.0, 0.5]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case2_batch_grayscale",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.zeros((1, 100, 200, 1), dtype=np.float16)
    boxes = np.array([[
        [0.05, 0.1, 0.3, 0.4],
        [0.4, 0.2, 0.9, 0.8],
        [0.0, 0.0, 1.0, 1.0]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case3_large_float16",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.randn(3, 10, 10, 4)).astype(np.float32)
    boxes = np.zeros((3, 0, 4), dtype=np.float32)
    input_dict = {
        "name": "case4_no_boxes",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.randn(1, 8, 8, 3)).astype(np.float32)
    boxes = np.array([[
        [-0.1, -0.1, 1.2, 1.3],
        [0.3, 0.3, 0.7, 0.7]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case5_out_of_range_boxes",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.array([[[[0.25]]]], dtype=np.float32)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    input_dict = {
        "name": "case6_minimal",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(2, 4, 6, 3)).astype(np.float16)
    boxes = np.array([
        [[0.1, 0.1, 0.9, 0.9]],
        [[0.3, 0.0, 0.7, 1.0]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case7_float16_rgb",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.linspace(0, 1, 1 * 64 * 64 * 1, dtype=np.float32).reshape(1, 64, 64, 1)
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.01, 0.01, 0.99, 0.99],
        [0.25, 0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 0.5]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case8_many_boxes",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(4, 16, 16, 3)).astype(np.float32)
    boxes = np.array([
        [[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]],
        [[0.2, 0.2, 0.8, 0.7], [0.0, 0.0, 0.2, 0.3]],
        [[0.0, 0.8, 1.0, 1.0], [0.4, 0.4, 0.6, 0.6]],
        [[0.3, 0.1, 0.5, 0.9], [0.1, 0.7, 0.2, 0.9]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case9_batch4_rgb",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.randn(5, 3, 3, 4)).astype(np.float32)
    boxes = np.array([
        [[0.0, 0.0, 1.0, 1.0]],
        [[0.2, 0.2, 0.8, 0.8]],
        [[0.3, 0.1, 0.7, 0.9]],
        [[0.0, 0.4, 1.0, 0.6]],
        [[0.5, 0.0, 0.5, 1.0]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case10_depth4",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(1, 32, 48, 4)).astype(np.float32)
    boxes = np.array([[
        [0.1, 0.1, 0.1, 0.5],
        [0.2, 0.2, 0.8, 0.2],
        [0.0, 0.0, 1.0, 0.2],
        [0.3, 0.4, 0.9, 0.95]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case11_rgba_degenerate",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(2, 7, 3, 1)).astype(np.float16)
    boxes = np.array([
        [[0.0, 0.0, 0.5, 1.0], [0.2, 0.2, 0.6, 0.8], [0.6, 0.1, 1.0, 0.9]],
        [[0.1, 0.2, 0.9, 0.7], [0.0, 0.5, 0.4, 0.9], [0.3, 0.0, 0.7, 0.4]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case12_float16_small_wide",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = tf_raw_ops_DrawBoundingBoxes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxes'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DrawBoundingBoxes', generated_inputs['tf.raw_ops.DrawBoundingBoxes'], lib="tf", suffix=0)
