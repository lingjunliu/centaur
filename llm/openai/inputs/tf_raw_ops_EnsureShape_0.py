
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_EnsureShape_inputs():
    list_of_inputs = []

    arr = np.array(7, dtype=np.int32)
    shape = []
    input_dict = {"name": "scalar_int32_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0.1, -0.2, 3.3, 4.4, -5.5], dtype=np.float32)
    shape = [-1]
    input_dict = {"name": "vector_float32_dynamic", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1, 2, 3], [-4, -5, -6]], dtype=np.int64)
    shape = [2, 3]
    input_dict = {"name": "matrix_int64_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1, -1, 2, -2], [3, -3, 4, -4]], dtype=np.int32)
    shape = [2, -1]
    input_dict = {"name": "matrix_partial_unknown", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(2*3*4, dtype=np.float64).reshape(2, 3, 4)
    shape = [-1, 3, 4]
    input_dict = {"name": "tensor3d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([True, False, True], dtype=bool)
    shape = [3]
    input_dict = {"name": "vector_bool_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1+2j, -3+4j], [5-6j, -7-8j]], dtype=np.complex64)
    shape = [2, 2]
    input_dict = {"name": "matrix_complex64_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.int32)
    shape = [0]
    input_dict = {"name": "empty_vector_int32", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((0, 3), dtype=np.float32)
    shape = [0, 3]
    input_dict = {"name": "zero_rows_matrix_float32", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(1*2*3*4, dtype=np.float16).reshape(1, 2, 3, 4)
    shape = [1, 2, 3, 4]
    input_dict = {"name": "tensor4d_float16_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([b"hello", b"world"], dtype=np.object_)
    shape = [2]
    input_dict = {"name": "vector_string_bytes", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((1, 0, 5), dtype=np.int8)
    shape = [1, 0, 5]
    input_dict = {"name": "tensor3d_zero_mid_dim", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+0j, -2+3j, 4-5j, -6-7j], dtype=np.complex128)
    shape = [-1]
    input_dict = {"name": "vector_complex128_dynamic", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_EnsureShape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
