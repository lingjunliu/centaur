
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_CropAndResizeGradBoxes_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    grads = np.random.rand(1, 10, 10, 3).astype(np.float32)
    image = np.random.rand(1, 20, 20, 3).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple boxes
    grads = np.random.rand(2, 5, 5, 1).astype(np.float32)
    image = np.random.rand(1, 15, 15, 1).astype(np.float32)
    boxes = np.array([[0.2, 0.2, 0.6, 0.6], [0.4, 0.4, 0.8, 0.8]]).astype(np.float32)
    box_ind = np.array([0, 0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different image size
    grads = np.random.rand(1, 8, 8, 3).astype(np.float32)
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple images
    grads = np.random.rand(3, 4, 4, 2).astype(np.float32)
    image = np.random.rand(2, 16, 16, 2).astype(np.float32)
    boxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]]).astype(np.float32)
    box_ind = np.array([0, 1, 0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: grads with different depth
    grads = np.random.rand(1, 6, 6, 5).astype(np.float32)
    image = np.random.rand(1, 24, 24, 5).astype(np.float32)
    boxes = np.array([[0.4, 0.4, 0.6, 0.6]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: flipped boxes
    grads = np.random.rand(1, 7, 7, 3).astype(np.float32)
    image = np.random.rand(1, 28, 28, 3).astype(np.float32)
    boxes = np.array([[0.8, 0.8, 0.2, 0.2]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boxes outside range
    grads = np.random.rand(1, 9, 9, 4).astype(np.float32)
    image = np.random.rand(1, 36, 36, 4).astype(np.float32)
    boxes = np.array([[-0.1, -0.1, 1.1, 1.1]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: float64 image
    grads = np.random.rand(1, 10, 10, 3).astype(np.float32)
    image = np.random.rand(1, 20, 20, 3).astype(np.float64)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 image
    grads = np.random.rand(1, 10, 10, 3).astype(np.float32)
    image = np.random.randint(0, 256, size=(1, 20, 20, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half image
    grads = np.random.rand(1, 10, 10, 3).astype(np.float32)
    image = np.random.rand(1, 20, 20, 3).astype(np.float16)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]]).astype(np.float32)
    box_ind = np.array([0]).astype(np.int32)
    method = "bilinear"
    name = None

    input_dict = {
        "grads": grads,
        "image": image,
        "boxes": boxes,
        "box_ind": box_ind,
        "method": method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CropAndResizeGradBoxes"] = tf_raw_ops_CropAndResizeGradBoxes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CropAndResizeGradBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CropAndResizeGradBoxes'.")

check_valid('tf.raw_ops.CropAndResizeGradBoxes', generated_inputs['tf.raw_ops.CropAndResizeGradBoxes'], lib="tf", suffix=0)
