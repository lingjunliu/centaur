
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_get_static_value_inputs():
    list_of_inputs = []

    tensor = tf.constant(np.int32(10))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.float32(-3.5))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([[1.0, -2.5], [3.1, 4.2]], dtype=np.float64))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([True, False, True], dtype=np.bool_))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([1+2j, -3+0.5j], dtype=np.complex64))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    b = tf.constant(np.array([-1, 0, 1], dtype=np.int32))
    tensor = tf.add(a, b)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = tf.constant(np.arange(6, dtype=np.int32))
    shape = tf.constant(np.array([2, 3], dtype=np.int32))
    tensor = tf.reshape(base, shape)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    mat = tf.constant(np.arange(12, dtype=np.float32).reshape(3, 4))
    tensor = tf.transpose(mat)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t1 = tf.constant(np.array([[1, 2]], dtype=np.int32))
    t2 = tf.constant(np.array([[3, 4]], dtype=np.int32))
    tensor = tf.concat([t1, t2], axis=0)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([0, 255], dtype=np.uint8))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base_int = tf.constant(np.array([1, 0, 1], dtype=np.int32))
    tensor = tf.cast(base_int, tf.bool)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    mat = tf.constant(np.arange(6, dtype=np.int32).reshape(2, 3))
    axis = tf.constant(np.int32(1))
    tensor = tf.reduce_sum(mat, axis=axis)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.add(tf.constant(np.int32(3)), tf.Variable(np.int32(4)))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([], dtype=np.float32))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.get_static_value"] = tf_get_static_value_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.get_static_value' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.get_static_value'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.get_static_value', generated_inputs['tf.get_static_value'], lib="tf", suffix=0)
