
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
rs = np.random.RandomState(0)

def tf_signal_rfft2d_inputs():
    list_of_inputs = []

    input_tensor = rs.randn(4, 6).astype(np.float32)
    fft_length = [4, 6]
    name = "case_exact_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.uniform(-5, 5, size=(5, 7)).astype(np.float32)
    fft_length = [4, 6]
    name = "case_crop_both_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(3, 3).astype(np.float32)
    fft_length = [5, 4]
    name = "case_pad_both_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(2, 8, 5).astype(np.float32)
    fft_length = [8, 5]
    name = "case_batch_exact_3d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(3, 4, 4).astype(np.float32)
    fft_length = [2, 6]
    name = "case_batch_crop_pad_3d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(2, 3, 6, 10).astype(np.float32)
    fft_length = [6, 10]
    name = "case_4d_exact_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(1, 2, 7, 5).astype(np.float64)
    fft_length = [8, 8]
    name = "case_4d_pad_both_f64"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(2, 2, 2, 4, 4).astype(np.float32)
    fft_length = [4, 4]
    name = "case_5d_exact_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = np.linspace(-1.0, 1.0, num=8, dtype=np.float32).reshape(1, 8)
    fft_length = [1, 8]
    name = "case_degenerate_row_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = np.full((10, 1, 1), 3.14, dtype=np.float64)
    fft_length = [1, 1]
    name = "case_degenerate_inner_3d_f64"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(9, 1).astype(np.float64)
    fft_length = [16, 1]
    name = "case_pad_rows_2d_f64"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.uniform(-1, 1, size=(5, 2, 9)).astype(np.float32)
    fft_length = [5, 7]
    name = "case_batch_pad_crop_3d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    return list_of_inputs

generated_inputs["tf.signal.rfft2d"] = tf_signal_rfft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.rfft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.rfft2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.rfft2d', generated_inputs['tf.signal.rfft2d'], lib="tf", suffix=0)
