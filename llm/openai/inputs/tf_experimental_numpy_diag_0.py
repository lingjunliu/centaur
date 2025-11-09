
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []

    v = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.int64)
    k = 1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1.5, -2.0, 3.5, 4.0],
                  [0.0, 7.25, 8.5, -9.0]], dtype=np.float64)
    k = -1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([], dtype=np.float32)
    k = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.zeros((0, 0), dtype=bool)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([1+2j, -3+0.5j, -1j, 4+4j], dtype=np.complex64)
    k = 2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    real = np.arange(20, dtype=np.float64).reshape(4, 5)
    imag = np.arange(20, dtype=np.float64).reshape(4, 5)
    v = (real + 1j * imag).astype(np.complex128)
    k = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([True, False, True, True, False], dtype=bool)
    k = -2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1, 2, 3, 4, 5]], dtype=np.int16)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[10], [20], [30], [40], [50]], dtype=np.int16)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]], dtype=np.float16)
    k = -3
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([255, 0, 128], dtype=np.uint8)
    k = -1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    rng = np.random.RandomState(0)
    v = rng.randn(6, 3).astype(np.float32)
    k = 4
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[np.nan, np.inf],
                  [-np.inf, -0.0]], dtype=np.float32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.empty((0, 3), dtype=np.float64)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.diag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.diag', generated_inputs['tf.experimental.numpy.diag'], lib="tf", suffix=0)
