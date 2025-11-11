
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_ensure_shape_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape1 = [2, 2]
    name1 = "ensure_shape_1"
    input_dict1 = {"name": name1, "input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shape2 = [5]
    name2 = "ensure_shape_2"
    input_dict2 = {"name": name2, "input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    shape3 = [2, 2, 3]
    name3 = "ensure_shape_3"
    input_dict3 = {"name": name3, "input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape4 = [2, 2]
    name4 = "ensure_shape_4"
    input_dict4 = {"name": name4, "input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1], dtype=np.int32)
    shape5 = [1]
    name5 = "ensure_shape_5"
    input_dict5 = {"name": name5, "input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3], dtype=np.float64)
    shape6 = [3]
    name6 = "ensure_shape_6"
    input_dict6 = {"name": name6, "input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    shape7 = [2, 2, 2]
    name7 = "ensure_shape_7"
    input_dict7 = {"name": name7, "input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1, 2, 3, 4], dtype=np.int16)
    shape8 = [4]
    name8 = "ensure_shape_8"
    input_dict8 = {"name": name8, "input": input8, "shape": shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    shape9 = [2,3]
    name9 = "ensure_shape_9"
    input_dict9 = {"name": name9, "input": input9, "shape": shape9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([10], dtype=np.int8)
    shape10 = [1]
    name10 = "ensure_shape_10"
    input_dict10 = {"name": name10, "input": input10, "shape": shape10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_ensure_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
