
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_lstsq_inputs():
    list_of_inputs = []

    matrix1 = np.random.rand(2, 3).astype(np.float64)
    rhs1 = np.random.rand(2, 1).astype(np.float64)
    input_dict1 = {
        "matrix": matrix1,
        "rhs": rhs1,
        "l2_regularizer": 0.0,
        "fast": True,
        "name": "lstsq_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    matrix2 = np.random.rand(3, 2).astype(np.float64)
    rhs2 = np.random.rand(3, 1).astype(np.float64)
    input_dict2 = {
        "matrix": matrix2,
        "rhs": rhs2,
        "l2_regularizer": 1e-6,
        "fast": False,
        "name": "lstsq_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    matrix3 = np.random.rand(4, 4).astype(np.float64)
    rhs3 = np.random.rand(4, 2).astype(np.float64)
    input_dict3 = {
        "matrix": matrix3,
        "rhs": rhs3,
        "l2_regularizer": 0.1,
        "fast": True,
        "name": "lstsq_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    matrix4 = np.random.rand(5, 3).astype(np.float64)
    rhs4 = np.random.rand(5, 3).astype(np.float64)
    input_dict4 = {
        "matrix": matrix4,
        "rhs": rhs4,
        "l2_regularizer": 0.0,
        "fast": False,
        "name": "lstsq_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    matrix5 = np.random.rand(2, 2).astype(np.float64)
    rhs5 = np.random.rand(2, 1).astype(np.float64)
    input_dict5 = {
        "matrix": matrix5,
        "rhs": rhs5,
        "l2_regularizer": 1.0,
        "fast": True,
        "name": "lstsq_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    matrix6 = np.random.rand(3, 4).astype(np.float64)
    rhs6 = np.random.rand(3, 2).astype(np.float64)
    input_dict6 = {
        "matrix": matrix6,
        "rhs": rhs6,
        "l2_regularizer": 0.01,
        "fast": False,
        "name": "lstsq_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    matrix7 = np.random.rand(4, 3).astype(np.float64)
    rhs7 = np.random.rand(4, 1).astype(np.float64)
    input_dict7 = {
        "matrix": matrix7,
        "rhs": rhs7,
        "l2_regularizer": 0.5,
        "fast": True,
        "name": "lstsq_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    matrix8 = np.random.rand(5, 5).astype(np.float64)
    rhs8 = np.random.rand(5, 3).astype(np.float64)
    input_dict8 = {
        "matrix": matrix8,
        "rhs": rhs8,
        "l2_regularizer": 0.0,
        "fast": False,
        "name": "lstsq_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    matrix9 = np.random.rand(2, 4).astype(np.float64)
    rhs9 = np.random.rand(2, 2).astype(np.float64)
    input_dict9 = {
        "matrix": matrix9,
        "rhs": rhs9,
        "l2_regularizer": 1e-8,
        "fast": True,
        "name": "lstsq_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    matrix10 = np.random.rand(6, 2).astype(np.float64)
    rhs10 = np.random.rand(6, 3).astype(np.float64)
    input_dict10 = {
        "matrix": matrix10,
        "rhs": rhs10,
        "l2_regularizer": 0.2,
        "fast": False,
        "name": "lstsq_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.linalg.lstsq"] = tf_linalg_lstsq_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lstsq'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.lstsq', generated_inputs['tf.linalg.lstsq'], lib="tf", suffix=0)
