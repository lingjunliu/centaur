
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_identity_inputs():
    list_of_inputs = []

    arr = np.array([-3, 0, 7, -1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "id_int32_1d", "input": arr}))

    arr = np.array([[1.5, -2.3], [np.nan, np.inf]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "id_float32_2d_nan_inf", "input": arr}))

    arr = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"name": "id_bool_3d", "input": arr}))

    arr = np.array([[1+2j, -3-0j, 0+0j], [4-5j, -6+7j, 8+0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"name": "id_complex64_2x3", "input": arr}))

    arr = np.array([b"a", b"bb", b"ccc", b""], dtype=object)
    list_of_inputs.append(copy.deepcopy({"name": "id_bytes_string_1d", "input": arr}))

    arr = np.array(-0.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"name": "id_float64_scalar_negzero", "input": arr}))

    arr = np.array([[0, 255, 128], [64, 32, 16]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"name": "id_uint8_2d", "input": arr}))

    arr = np.array([[[[1, -1, 2]], [[3, -3, 4]]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "id_int64_4d", "input": arr}))

    arr = np.array([[-1.5, 0.0, 2.25], [3.5, -4.75, 5.125]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "id_float16_2d", "input": arr}))

    arr = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "id_empty_int32_1d", "input": arr}))

    arr = np.empty((2, 0), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "id_empty_float32_2x0", "input": arr}))

    arr = np.array([np.nan + 1j, 2 - np.inf*1j, -3 + 0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"name": "id_complex128_1d_nan_inf", "input": arr}))

    arr = np.array([["hello", "世界"], ["🌟", ""]], dtype=object)
    list_of_inputs.append(copy.deepcopy({"name": "id_unicode_string_2d", "input": arr}))

    arr = np.array(True, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"name": "id_bool_scalar", "input": arr}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Identity"] = tf_raw_ops_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Identity', generated_inputs['tf.raw_ops.Identity'], lib="tf", suffix=0)
