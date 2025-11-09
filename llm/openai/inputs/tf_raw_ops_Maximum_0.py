
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []

    x = np.array([0., -1., 2., -3., 4.], dtype=np.float32)
    y = np.array([-2., 0.5, 1.5, -5., 4.], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "max_f32_vec", "x": x, "y": y}))

    x = np.array([[-5.0, 0.0, 7.5], [1.2, -3.4, 9.9]], dtype=np.float64)
    y = np.array(-3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"name": "max_f64_mat_scalar", "x": x, "y": y}))

    x = np.arange(-12, 12, dtype=np.float16).reshape(2, 3, 4)
    y = np.array([[[0.5], [-1.0], [3.0]]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "max_f16_3d_bcast", "x": x, "y": y}))

    x = np.array([[-10, 0, 5], [7, -8, 2]], dtype=np.int32)
    y = np.array([[-5, -1, 6], [7, 8, -3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "max_i32_2d_same", "x": x, "y": y}))

    x = np.array([-128, -1, 0, 127], dtype=np.int8)
    y = np.array([-2], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"name": "max_i8_vec_scalar", "x": x, "y": y}))

    x = np.array([[[0], [5], [255]], [[100], [150], [200]]], dtype=np.uint8)
    y = np.array([[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"name": "max_u8_3d_bcast", "x": x, "y": y}))

    x = np.array(
        [
            [[-32768, -123, 0], [32767, 1000, -1]],
            [[-500, 500, -200], [200, -100, 100]],
        ],
        dtype=np.int16,
    ).reshape(2, 1, 2, 3)
    y = np.array(-1000, dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"name": "max_i16_4d_scalar", "x": x, "y": y}))

    x = np.array(-1234567890123456789, dtype=np.int64)
    y = np.array([10, -20], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "max_i64_scalar_vec", "x": x, "y": y}))

    x = np.array([[np.nan, np.inf], [-np.inf, 3.0]], dtype=np.float32)
    y = np.array([[1.0, -np.inf], [np.inf, np.nan]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "max_f32_nans_infs", "x": x, "y": y}))

    x = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    y = np.array([[[0.0], [1.5], [2.5], [3.5]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "max_f32_bcast_mixed", "x": x, "y": y}))

    x = np.array([np.iinfo(np.int32).min, -1000000000, 0, 1000000000, np.iinfo(np.int32).max], dtype=np.int32)
    y = np.array([-1, -2000000000, 1, 2000000000, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "max_i32_large_values", "x": x, "y": y}))

    x = np.array([[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]], dtype=np.float16)
    y = np.array([0.0, -3.0, 2.0, -9.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "max_f16_matrix_vector", "x": x, "y": y}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Maximum"] = tf_raw_ops_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Maximum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Maximum', generated_inputs['tf.raw_ops.Maximum'], lib="tf", suffix=0)
