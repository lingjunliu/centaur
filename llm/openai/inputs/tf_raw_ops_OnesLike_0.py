
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_OnesLike_inputs():
    list_of_inputs = []

    x = np.array([[True, False], [False, True]], dtype=bool)
    name = "bool_2x2"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-5, 0, 7], dtype=np.int8)
    name = "int8_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([0, 127, 255], dtype=np.uint8)
    name = "uint8_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[-12345, 0, 12345], [32767, -32768, 42]], dtype=np.int16)
    name = "int16_2x3"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((0, 3), dtype=np.int32)
    name = "int32_empty_0x3"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((2, 0, 4), dtype=np.int64)
    name = "int64_2x0x4"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-1.5, 0.0, 3.25, 7.75], dtype=np.float16)
    name = "float16_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[1.0, -2.0], [3.5, 4.25]], [[-5.75, 6.125], [0.0, -0.5]]], dtype=np.float32)
    name = "float32_2x2x2"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(-3.141592653589793, dtype=np.float64)
    name = "float64_scalar"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1+2j, -3-4j, 0+0j], dtype=np.complex64)
    name = "complex64_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(-42, dtype=np.int32)
    name = "int32_scalar_neg"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((0,), dtype=np.float32)
    name = "float32_empty_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.OnesLike"] = tf_raw_ops_OnesLike_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OnesLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OnesLike'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.OnesLike', generated_inputs['tf.raw_ops.OnesLike'], lib="tf", suffix=0)
