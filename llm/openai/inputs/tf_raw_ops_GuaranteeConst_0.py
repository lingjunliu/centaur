
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
np.random.seed(42)
tf.random.set_seed(42)

def tf_raw_ops_GuaranteeConst_inputs():
    list_of_inputs = []

    arr = np.array(5, dtype=np.int32)
    input_dict = {"name": "gc_scalar_int32", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    input_dict = {"name": "gc_1d_float32_neg_pos", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    input_dict = {"name": "gc_2d_int64_matrix", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.random.randn(2, 2, 3).astype(np.float64)
    input_dict = {"name": "gc_3d_float64_random", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {"name": "gc_2d_bool", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+2j, -3+0.5j], dtype=np.complex64)
    input_dict = {"name": "gc_1d_complex64", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0, 255, 128], dtype=np.uint8)
    input_dict = {"name": "gc_1d_uint8", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([b"hello", b"world"], dtype=object)
    input_dict = {"name": "gc_1d_bytes_strings", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.float32)
    input_dict = {"name": "gc_empty_float32", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = (np.random.randint(-1000, 1000, size=(2, 3, 4, 5))).astype(np.int16)
    input_dict = {"name": "gc_4d_int16_random", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    input_dict = {"name": "gc_special_float64", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.array([[1.0, -2.0], [3.5, 0.0]], dtype=np.float64)
    imag = np.array([[0.5, 1.5], [-4.0, 2.25]], dtype=np.float64)
    arr = real + 1j * imag
    arr = arr.astype(np.complex128)
    input_dict = {"name": "gc_2d_complex128", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(["こんにちは", "世界"], dtype=object)
    input_dict = {"name": "gc_unicode_strings", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([2147483647, -2147483647], dtype=np.int32)
    input_dict = {"name": "gc_edge_int32_values", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GuaranteeConst"] = tf_raw_ops_GuaranteeConst_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GuaranteeConst' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GuaranteeConst'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.GuaranteeConst', generated_inputs['tf.raw_ops.GuaranteeConst'], lib="tf", suffix=0)
