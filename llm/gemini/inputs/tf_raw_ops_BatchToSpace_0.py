
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_batchtospace_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input1 = np.arange(1 * 2 * 2 * 3).reshape(1 * 2 * 2, 1, 1, 3).astype(np.float32)
    crops1 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size1 = 2
    input_dict1 = {"input": input1, "crops": crops1, "block_size": block_size1, "name": "batch_to_space_1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Cropping
    input2 = np.arange(4 * 2 * 2 * 1).reshape(4 * 2 * 2, 1, 1, 1).astype(np.float32)
    crops2 = np.array([[1, 1], [1, 1]]).astype(np.int32)
    block_size2 = 2
    input_dict2 = {"input": input2, "crops": crops2, "block_size": block_size2, "name": "batch_to_space_2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different depth
    input3 = np.arange(1 * 2 * 2 * 5).reshape(1 * 2 * 2, 1, 1, 5).astype(np.float32)
    crops3 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size3 = 2
    input_dict3 = {"input": input3, "crops": crops3, "block_size": block_size3, "name": "batch_to_space_3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger input
    input4 = np.arange(4 * 3 * 3 * 2).reshape(4 * 3 * 3, 1, 1, 2).astype(np.float32)
    crops4 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size4 = 3
    input_dict4 = {"input": input4, "crops": crops4, "block_size": block_size4, "name": "batch_to_space_4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Int64 crops
    input5 = np.arange(1 * 2 * 2 * 3).reshape(1 * 2 * 2, 1, 1, 3).astype(np.float32)
    crops5 = np.array([[0, 0], [0, 0]]).astype(np.int64)
    block_size5 = 2
    input_dict5 = {"input": input5, "crops": crops5, "block_size": block_size5, "name": "batch_to_space_5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Non-zero crops
    input6 = np.arange(4 * 2 * 2 * 1).reshape(4 * 2 * 2, 1, 1, 1).astype(np.float32)
    crops6 = np.array([[0, 1], [1, 0]]).astype(np.int32)
    block_size6 = 2
    input_dict6 = {"input": input6, "crops": crops6, "block_size": block_size6, "name": "batch_to_space_6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: block_size = 3
    input7 = np.arange(1 * 3 * 3 * 2).reshape(1 * 3 * 3, 1, 1, 2).astype(np.float32)
    crops7 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size7 = 3
    input_dict7 = {"input": input7, "crops": crops7, "block_size": block_size7, "name": "batch_to_space_7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Different height and width (1,2)
    input8 = np.arange(1 * 2 * 2 * 3).reshape(1 * 2 * 2, 1, 1, 3).astype(np.float32)
    crops8 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size8 = 2
    input_dict8 = {"input": input8, "crops": crops8, "block_size": block_size8, "name": "batch_to_space_8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.arange(4 * 2 * 2 * 1).reshape(4 * 2 * 2, 1, 1, 1).astype(np.float32)
    crops9 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size9 = 2
    input_dict9 = {"input": input9, "crops": crops9, "block_size": block_size9, "name": "batch_to_space_9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Different shape.
    input10 = np.arange(16 * 2 * 2 * 1).reshape(16 * 2 * 2, 1, 1, 1).astype(np.float32)
    crops10 = np.array([[0, 0], [0, 0]]).astype(np.int32)
    block_size10 = 4
    input_dict10 = {"input": input10, "crops": crops10, "block_size": block_size10, "name": "batch_to_space_10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BatchToSpace"] = tf_raw_ops_batchtospace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchToSpace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchToSpace'.")

check_valid('tf.raw_ops.BatchToSpace', generated_inputs['tf.raw_ops.BatchToSpace'], lib="tf", suffix=0)
