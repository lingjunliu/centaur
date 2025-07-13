
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_shape_n_inputs():
    list_of_inputs = []

    # Input 1
    input_list = [tf.constant(np.array([1, 2, 3]))]
    out_type = tf.int32
    name = "shape_op_1"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_list = [tf.constant(np.array([[1, 2], [3, 4]]))]
    out_type = tf.int64
    name = "shape_op_2"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_list = [tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))]
    out_type = tf.int32
    name = "shape_op_3"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_list = [tf.constant(np.array([1, 2, 3], dtype=np.float32))]
    out_type = tf.int32
    name = "shape_op_4"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_list = [tf.constant(np.array([1, 2, 3], dtype=np.int64))]
    out_type = tf.int64
    name = "shape_op_5"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_list = [tf.constant(np.array([1]))]
    out_type = tf.int32
    name = "shape_op_6"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_list = [tf.constant(np.array([[1,2,3],[4,5,6],[7,8,9]]))]
    out_type = tf.int32
    name = "shape_op_7"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_list = [tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],[[9,10],[11,12]]]))]
    out_type = tf.int64
    name = "shape_op_8"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_list = [tf.constant(np.array([1, 2, 3])), tf.constant(np.array([4, 5, 6]))]
    out_type = tf.int32
    name = "shape_op_9"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_list = [tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])), tf.constant(np.array([9, 10, 11]))]
    out_type = tf.int64
    name = "shape_op_10"
    input_dict = {"input": input_list, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.shape_n"] = tf_shape_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.shape_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.shape_n'.")

check_valid('tf.shape_n', generated_inputs['tf.shape_n'], lib="tf", suffix=0)
