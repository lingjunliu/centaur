
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_prod_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis1 = np.array([0], dtype=np.int32)
    keep_dims1 = True
    name1 = "prod_1"
    input_dict1 = {
        'keep_dims': keep_dims1,
        'name': name1,
        'input': input1,
        'axis': axis1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3, 4], dtype=np.int32)
    axis2 = np.array([0], dtype=np.int64)
    keep_dims2 = False
    name2 = "prod_2"
    input_dict2 = {
        'keep_dims': keep_dims2,
        'name': name2,
        'input': input2,
        'axis': axis2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis3 = np.array([1, 2], dtype=np.int32)
    keep_dims3 = True
    name3 = "prod_3"
    input_dict3 = {
        'keep_dims': keep_dims3,
        'name': name3,
        'input': input3,
        'axis': axis3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    axis4 = np.array([0], dtype=np.int64)
    keep_dims4 = False
    name4 = "prod_4"
    input_dict4 = {
        'keep_dims': keep_dims4,
        'name': name4,
        'input': input4,
        'axis': axis4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3], dtype=np.uint8)
    axis5 = np.array([0], dtype=np.int32)
    keep_dims5 = False
    name5 = "prod_5"
    input_dict5 = {
        'keep_dims': keep_dims5,
        'name': name5,
        'input': input5,
        'axis': axis5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]], dtype=np.complex64)
    axis6 = np.array([1], dtype=np.int32)
    keep_dims6 = True
    name6 = "prod_6"
    input_dict6 = {
        'keep_dims': keep_dims6,
        'name': name6,
        'input': input6,
        'axis': axis6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis7 = np.array([0], dtype=np.int64)
    keep_dims7 = False
    name7 = "prod_7"
    input_dict7 = {
        'keep_dims': keep_dims7,
        'name': name7,
        'input': input7,
        'axis': axis7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    axis8 = np.array([1], dtype=np.int32)
    keep_dims8 = False
    name8 = "prod_8"
    input_dict8 = {
        'keep_dims': keep_dims8,
        'name': name8,
        'input': input8,
        'axis': axis8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis9 = np.array([0, 1], dtype=np.int32)
    keep_dims9 = True
    name9 = "prod_9"
    input_dict9 = {
        'keep_dims': keep_dims9,
        'name': name9,
        'input': input9,
        'axis': axis9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis10 = np.array([0], dtype=np.int32)
    keep_dims10 = True
    name10 = "prod_10"
    input_dict10 = {
        'keep_dims': keep_dims10,
        'name': name10,
        'input': input10,
        'axis': axis10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Prod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Prod', generated_inputs['tf.raw_ops.Prod'], lib="tf", suffix=0)
