
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sin_inputs():
    list_of_inputs = []

    x = np.array([-np.inf, -9.0, -0.5, 1.0, 1.2, 200.0, 10.0, np.inf], dtype=np.float32)
    name = "sin_vec_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(0.0, dtype=np.float64)
    name = "sin_scalar_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[-3.0, -1.5, -0.0], [0.5, 1.0, 3.0]], dtype=np.float16)
    name = "sin_mat_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[-np.pi], [-np.pi/2], [0.0]], [[np.pi/2], [np.pi], [3.14]]], dtype=np.float32)
    name = "sin_3d_angles_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1+2j, -3-4j, 0+1j, -2+0j], dtype=np.complex64)
    name = "sin_complex64_vec"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[1e-3 - 2e-3j, -5.5 + 0.0j], [0.0 + 3.14159j, -2.0 - 1.0j]], dtype=np.complex128)
    name = "sin_complex128_mat"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([np.nan, 1e20, -1e-20, -7.0, 7.0], dtype=np.float64)
    name = "sin_specials_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float32).reshape(2,1,3,2)
    name = "sin_4d_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([], dtype=np.float32)
    name = "sin_empty_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((2, 0, 3), dtype=np.float64)
    name = "sin_zerosize_axis_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1e-4, 1e-5, -1e-5, -2e-4], dtype=np.float16)
    name = "sin_small_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[0.0+0.0j, 1.0+0.0j], [0.0+1.0j, -1.0+0.0j]]], dtype=np.complex64)
    name = "sin_complex64_3d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sin"] = tf_raw_ops_sin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sin'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Sin', generated_inputs['tf.raw_ops.Sin'], lib="tf", suffix=0)
