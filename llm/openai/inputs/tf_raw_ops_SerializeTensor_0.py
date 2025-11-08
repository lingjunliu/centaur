
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_raw_ops_serialize_tensor_inputs():
    list_of_inputs = []

    tensor = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    input_dict = {"name": "serialize_case_int32_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[0.1, np.nan, -np.inf], [np.inf, -3.5, 0.0]],
                       [[1.2, -2.3, 4.5], [6.7, -8.9, 10.11]]], dtype=np.float32)
    input_dict = {"name": "serialize_case_float32_3d_nan_inf", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([True, False, True, True, False], dtype=bool)
    input_dict = {"name": "serialize_case_bool_1d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array(np.pi, dtype=np.float64)
    input_dict = {"name": "serialize_case_float64_scalar", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1+2j, -3-4j], [5-6j, -7+8j]], dtype=np.complex64)
    input_dict = {"name": "serialize_case_complex64_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([1+0j, 0-1j, -2+3j, 4-5j], dtype=np.complex128)
    input_dict = {"name": "serialize_case_complex128_1d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randint(0, 256, size=(2, 3, 4, 1), dtype=np.uint8)
    input_dict = {"name": "serialize_case_uint8_4d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.empty((0, 3), dtype=np.int64)
    input_dict = {"name": "serialize_case_int64_empty_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[-1.5, 2.25, -3.75], [4.5, -5.125, 6.0]], dtype=np.float16)
    input_dict = {"name": "serialize_case_float16_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[[[1, -1, 2]]], [[[3, -3, 4]]]]], dtype=np.int8)
    input_dict = {"name": "serialize_case_int8_5d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SerializeTensor"] = tf_raw_ops_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SerializeTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SerializeTensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SerializeTensor', generated_inputs['tf.raw_ops.SerializeTensor'], lib="tf", suffix=0)
