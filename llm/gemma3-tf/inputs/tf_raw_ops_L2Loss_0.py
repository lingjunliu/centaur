
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_L2Loss_inputs():
    list_of_inputs = []

    input_dict = {
        "name": "L2Loss_1",
        "t": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_2",
        "t": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_3",
        "t": np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_4",
        "t": np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_5",
        "t": np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_6",
        "t": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_7",
        "t": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_8",
        "t": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_9",
        "t": np.array([0.0, 0.0, 0.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_L2Loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.L2Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.L2Loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.L2Loss', generated_inputs['tf.raw_ops.L2Loss'], lib="tf", suffix=0)
