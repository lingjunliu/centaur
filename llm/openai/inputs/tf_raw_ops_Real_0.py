
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import torch
import copy

def tf_raw_ops_real_inputs():
    list_of_inputs = []

    x = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_vec_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1 + 2j, -3 - 4j], [0 + 0j, 5 - 6j]], dtype=np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_matrix_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    imag = (-real - 0.5).astype(np.float32)
    x = (real + 1j * imag).astype(np.complex64)
    input_dict = {"Tout": np.dtype("float32"), "name": "real_3d_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(7.5 - 1.5j, dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_scalar_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_empty1d_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.linspace(-10, 10, num=24, dtype=np.float32).reshape(1, 2, 3, 4)
    imag = np.linspace(10, -10, num=24, dtype=np.float32).reshape(1, 2, 3, 4)
    x = (real + 1j * imag).astype(np.complex64)
    input_dict = {"Tout": np.dtype("float32"), "name": "real_4d_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base_real = np.arange(12, dtype=np.float64).reshape(3, 4)
    base_imag = np.flip(base_real, axis=1)
    x_full = (base_real + 1j * base_imag).astype(np.complex128)
    x = x_full[::2, ::2]
    input_dict = {"Tout": np.float64, "name": "real_noncontig_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-30 + 1e-30j, -1e-40 + 2e-40j], dtype=np.complex128)
    input_dict = {"Tout": np.dtype("float64"), "name": "real_tinyvals_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e10 - 1e9j, -3.4e20 + 1e19j], dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_largevals_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan + 1j, np.inf - np.inf * 1j, -np.inf + (np.nan * 1j)], dtype=np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_nan_inf_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.zeros((2, 0, 3), dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_empty_axes_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-0.0 + 0.0j]], dtype=np.complex128)
    input_dict = {"Tout": np.dtype("float64"), "name": "real_singleton2d_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([3 + 4j], dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_len1_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.array([[1e-6, -2e-6, 3e-6]], dtype=np.float64)
    imag = np.array([[4e-6, -5e-6, 6e-6]], dtype=np.float64)
    x = (real + 1j * imag).astype(np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_small_2d_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_raw_ops_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Real'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Real', generated_inputs['tf.raw_ops.Real'], lib="tf", suffix=0)
