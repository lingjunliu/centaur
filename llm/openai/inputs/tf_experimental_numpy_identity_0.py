
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_identity_inputs():
    list_of_inputs = []

    n = 0
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 1
    dtype = np.int32
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 2
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 3
    dtype = np.bool_
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 4
    dtype = np.complex64
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 5
    dtype = np.complex128
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 6
    dtype = np.uint8
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 7
    dtype = np.int64
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 8
    dtype = np.float16
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 9
    dtype = np.dtype("uint16")
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 10
    dtype = np.dtype("uint32")
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 11
    dtype = np.dtype("int8")
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.identity"] = tf_experimental_numpy_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.identity', generated_inputs['tf.experimental.numpy.identity'], lib="tf", suffix=0)
