
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import torch
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_relu_inputs():
    list_of_inputs = []

    features = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"name": "relu_f32_1D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.2, 2.5], [0.0, -3.4]], dtype=np.float64)
    input_dict = {"name": "relu_f64_2D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[-3, -2, -1], [0, 1, 2]], [[3, -4, 5], [-6, 7, -8]]], dtype=np.int32)
    input_dict = {"name": "relu_i32_3D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0, 1, 255], dtype=np.uint8)
    input_dict = {"name": "relu_u8_1D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-32768, -1, 0, 1, 32767]], dtype=np.int16)
    input_dict = {"name": "relu_i16_2D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(-5, dtype=np.int8)
    input_dict = {"name": "relu_i8_scalar_0D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[[-10, 0, 10]]], [[[20, -30, 40]]]], dtype=np.int64)
    input_dict = {"name": "relu_i64_4D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[-1.5, 2.0], [3.3, -4.4], [0.0, 5.5]],
                         [[-6.6, 7.7], [-8.8, 9.9], [10.0, -11.0]]], dtype=np.float16)
    input_dict = {"name": "relu_f16_3D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    input_dict = {"name": "relu_empty_f32", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[
        [
            [[-3.0, 0.0, 3.0]],
            [[4.5, -5.5, 6.5]]
        ],
        [
            [[7.0, -8.0, 9.0]],
            [[-1.0, 2.0, -3.0]]
        ]
    ]], dtype=np.float32)
    input_dict = {"name": "relu_f32_5D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1e6, -1.5, 0.0, 1.5, 3.4e5], dtype=np.float64)
    input_dict = {"name": "relu_f64_large_range", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0, 128, 255], [5, 10, 15]], dtype=np.uint8)
    input_dict = {"name": "relu_u8_2D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Relu"] = tf_raw_ops_relu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Relu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Relu', generated_inputs['tf.raw_ops.Relu'], lib="tf", suffix=0)
