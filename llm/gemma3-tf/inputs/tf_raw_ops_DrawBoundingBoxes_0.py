
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_DrawBoundingBoxes_inputs():
    list_of_inputs = []

    images1 = np.random.rand(2, 100, 200, 3).astype(np.float32)
    boxes1 = np.random.rand(2, 5, 4).astype(np.float32)
    name1 = "draw_boxes_1"
    input_dict1 = {'name': name1, 'images': images1, 'boxes': boxes1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    images2 = np.random.rand(1, 50, 100, 1).astype(np.float32)
    boxes2 = np.random.rand(1, 3, 4).astype(np.float32)
    name2 = "draw_boxes_2"
    input_dict2 = {'name': name2, 'images': images2, 'boxes': boxes2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    images3 = np.random.rand(4, 256, 256, 3).astype(np.float16)
    boxes3 = np.random.rand(4, 2, 4).astype(np.float32)
    name3 = "draw_boxes_3"
    input_dict3 = {'name': name3, 'images': images3, 'boxes': boxes3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    images4 = np.random.rand(1, 128, 128, 3).astype(np.float32)
    boxes4 = np.random.rand(1, 10, 4).astype(np.float32)
    name4 = "draw_boxes_4"
    input_dict4 = {'name': name4, 'images': images4, 'boxes': boxes4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    images5 = np.random.rand(3, 64, 64, 3).astype(np.float32)
    boxes5 = np.random.rand(3, 1, 4).astype(np.float32)
    name5 = "draw_boxes_5"
    input_dict5 = {'name': name5, 'images': images5, 'boxes': boxes5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    images6 = np.random.rand(2, 200, 300, 3).astype(np.float32)
    boxes6 = np.random.rand(2, 7, 4).astype(np.float32)
    name6 = "draw_boxes_6"
    input_dict6 = {'name': name6, 'images': images6, 'boxes': boxes6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    images7 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    boxes7 = np.random.rand(1, 4, 4).astype(np.float32)
    name7 = "draw_boxes_7"
    input_dict7 = {'name': name7, 'images': images7, 'boxes': boxes7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    images8 = np.random.rand(5, 100, 100, 3).astype(np.float32)
    boxes8 = np.random.rand(5, 6, 4).astype(np.float32)
    name8 = "draw_boxes_8"
    input_dict8 = {'name': name8, 'images': images8, 'boxes': boxes8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    images9 = np.random.rand(1, 224, 224, 3).astype(np.float32)
    boxes9 = np.random.rand(1, 8, 4).astype(np.float32)
    name9 = "draw_boxes_9"
    input_dict9 = {'name': name9, 'images': images9, 'boxes': boxes9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    images10 = np.random.rand(2, 150, 150, 3).astype(np.float32)
    boxes10 = np.random.rand(2, 3, 4).astype(np.float32)
    name10 = "draw_boxes_10"
    input_dict10 = {'name': name10, 'images': images10, 'boxes': boxes10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
