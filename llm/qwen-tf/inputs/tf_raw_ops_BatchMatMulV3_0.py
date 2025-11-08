
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_batch_matmul_v3_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    y = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": False,
        "grad_x": False,
        "grad_y": False,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    x = np.array([[[1., 2], [3., 4]], [[5., 6], [7., 8]]], dtype=np.float32)
    y = np.array([[[2., 3], [4., 5]], [[6., 7], [8., 9]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": False,
        "grad_x": False,
        "grad_y": False,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    y = np.array([[5., 6.], [7., 8.]], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": True,
        "adj_y": False,
        "grad_x": False,
        "grad_y": False,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    y = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": True,
        "grad_x": False,
        "grad_y": False,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    y = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": True,
        "adj_y": True,
        "grad_x": False,
        "grad_y": False,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    x = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    y = np.array([[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": False,
        "grad_x": True,
        "grad_y": False,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    x = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    y = np.array([[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": False,
        "grad_x": False,
        "grad_y": True,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    x = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    y = np.array([[13., 14., 15.], [16., 17., 18.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": False,
        "grad_x": False,
        "grad_y": False,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    y = np.array([[5., 6.], [7., 8.]], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": False,
        "adj_y": True,
        "grad_x": True,
        "grad_y": True,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    x = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    y = np.array([[13., 14., 15.], [16., 17., 18.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "adj_x": True,
        "adj_y": False,
        "grad_x": False,
        "grad_y": False,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.BatchMatMulV3"] = generate_batch_matmul_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BatchMatMulV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchMatMulV3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.BatchMatMulV3', generated_inputs['tf.raw_ops.BatchMatMulV3'], lib="tf", suffix=0)
