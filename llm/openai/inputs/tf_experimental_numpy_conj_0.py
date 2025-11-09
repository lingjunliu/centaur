
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_conj_inputs():
    list_of_inputs = []

    x = np.array([1+2j, -3+4j, -1j, 0+0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[-1.5, 2.0], [3.0, -4.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array(3+5j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    tmp = np.arange(12).reshape(4, 3).astype(np.complex128)
    tmp = tmp + 1j * tmp
    x = tmp[:, ::2]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array(
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
         [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
        dtype=np.uint8
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([np.inf + np.nan*1j, -np.inf - np.inf*1j, np.nan + 0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[0.0, -0.0], [1.5, -2.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.empty((2, 0, 3), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-2**40, 2**40 - 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([complex(0.0, -0.0), complex(-0.0, 0.0)], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.arange(10, dtype=np.float64)[::-1]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.conj"] = tf_experimental_numpy_conj_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.conj'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.conj', generated_inputs['tf.experimental.numpy.conj'], lib="tf", suffix=0)
