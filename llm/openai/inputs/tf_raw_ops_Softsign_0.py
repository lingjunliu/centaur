
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Softsign_inputs():
    rs = np.random.RandomState(0)
    rs2 = np.random.RandomState(123)
    list_of_inputs = []

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "basic_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.5, -2.5], [10.0, -0.0]], dtype=np.float64)
    name = "matrix_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(0.5, dtype=np.float32)
    name = "scalar_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    name = "empty_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.arange(-10, 10, dtype=np.float16)
    name = "range_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = rs.uniform(-5, 5, size=(2, 3, 4)).astype(np.float16)
    name = "rand3d_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.linspace(-100, 100, num=21, dtype=np.float64)[::3]
    name = "strided_view_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([np.inf, -np.inf, np.nan, 1.0, -1.0], dtype=np.float32)
    name = "nan_inf_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    name = "fortran_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1e-12, -1e-12, 1e12, -1e12], dtype=np.float64)
    name = "magnitude_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = rs2.normal(loc=0.0, scale=3.0, size=(2, 2, 2, 3)).astype(np.float32)
    name = "rand4d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.0, -7.5, 7.5, 15.0, -15.0]], dtype=np.float16)
    name = "row2d_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.123456789, -0.987654321], dtype=np.float64)
    name = "hi_precision_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.zeros((0, 3), dtype=np.float32)
    name = "empty_2d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softsign"] = tf_raw_ops_Softsign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softsign'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Softsign', generated_inputs['tf.raw_ops.Softsign'], lib="tf", suffix=0)
