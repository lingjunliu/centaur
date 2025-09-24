
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    tensor = np.array(1, dtype=np.int32)
    name = "scalar_int32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    name = "vector_float32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    name = "matrix_int64"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((2, 3, 4)).astype(np.float64)
    name = "tensor3d_float64"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[True, False], [False, True]], dtype=np.bool_)
    name = "bool_2x2"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([], dtype=np.float32)
    name = "empty_float32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.empty((2, 0, 3), dtype=np.float32)
    name = "zerosize_dim"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    name = "complex64_1d"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([np.nan, np.inf, -np.inf, 1e30], dtype=np.float64)
    name = "nan_inf_float64"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape(2, 3, 4, 5)
    name = "rank4_int32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = (np.arange(60, dtype=np.float16) - 30).reshape(3, 4, 5)
    name = "float16_3d"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_tensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.serialize_tensor', generated_inputs['tf.io.serialize_tensor'], lib="tf", suffix=0)
