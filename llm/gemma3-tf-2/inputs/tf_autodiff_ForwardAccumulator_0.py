
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_autodiff_forwardaccumulator_inputs():
    list_of_inputs = []
    primals1 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    tangents1 = tf.constant(np.array([[0.5], [0.2]], dtype=np.float32))
    x1 = tf.constant(1.0, dtype=np.float32)
    input_dict1 = {'primals': primals1, 'tangents': tangents1, 'x': x1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    primals2 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    tangents2 = tf.constant(np.array([0.1, -0.2, 0.3], dtype=np.float64))
    x2 = tf.constant(2.0, dtype=np.float64)
    input_dict2 = {'primals': primals2, 'tangents': tangents2, 'x': x2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    primals3 = tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.complex64))
    tangents3 = tf.constant(np.array([[0.1j], [-0.2j]], dtype=np.complex64))
    x3 = tf.constant(1.0 + 1.0j, dtype=np.complex64)
    input_dict3 = {'primals': primals3, 'tangents': tangents3, 'x': x3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    primals4 = tf.constant(np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    tangents4 = tf.constant(np.array([[1.0], [-1.0]], dtype=np.float32))
    x4 = tf.constant(-1.0, dtype=np.float32)
    input_dict4 = {'primals': primals4, 'tangents': tangents4, 'x': x4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    primals5 = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    tangents5 = tf.constant(np.array([[[0.5], [0.2]], [[0.1], [0.3]]], dtype=np.float32))
    x5 = tf.constant(2.0, dtype=np.float32)
    input_dict5 = {'primals': primals5, 'tangents': tangents5, 'x': x5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    primals6 = tf.constant(np.array([1.0, 2.0], dtype=np.float32))
    tangents6 = tf.constant(np.array([0.0, 1.0], dtype=np.float32))
    x6 = tf.constant(0.0, dtype=np.float32)
    input_dict6 = {'primals': primals6, 'tangents': tangents6, 'x': x6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    primals7 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16))
    tangents7 = tf.constant(np.array([[0.5], [0.2]], dtype=np.float16))
    x7 = tf.constant(1.0, dtype=np.float16)
    input_dict7 = {'primals': primals7, 'tangents': tangents7, 'x': x7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    primals8 = tf.constant(np.array([[-2.0, -3.0], [-4.0, -5.0]], dtype=np.float32))
    tangents8 = tf.constant(np.array([[0.1], [-0.2]], dtype=np.float32))
    x8 = tf.constant(-1.0, dtype=np.float32)
    input_dict8 = {'primals': primals8, 'tangents': tangents8, 'x': x8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    primals9 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    tangents9 = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    x9 = tf.constant(3.0, dtype=np.float32)
    input_dict9 = {'primals': primals9, 'tangents': tangents9, 'x': x9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    primals10 = tf.constant(np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    tangents10 = tf.constant(np.array([[1.0], [0.0]], dtype=np.float32))
    x10 = tf.constant(0.0, dtype=np.float32)
    input_dict10 = {'primals': primals10, 'tangents': tangents10, 'x': x10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_forwardaccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.autodiff.ForwardAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.autodiff.ForwardAccumulator'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.autodiff.ForwardAccumulator', generated_inputs['tf.autodiff.ForwardAccumulator'], lib="tf", suffix=0)
