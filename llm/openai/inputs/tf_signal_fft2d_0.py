
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fft2d_inputs():
    rs = np.random.RandomState(0)
    list_of_inputs = []

    arr1 = np.array([[1+2j, -3+4j], [0-1j, 2+0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr1, "name": "simple_2x2_c64"}))

    arr2 = np.array(
        [
            [1-1j, 2+3j, -4-5j, 6+0j, -7+8j],
            [9-10j, 0+0j, -1+2j, 3-4j, 5+6j],
            [-2-3j, 4+5j, -6+7j, 8-9j, 10+11j],
        ],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"input": arr2, "name": "rect_3x5_c128"}))

    arr3 = (rs.randn(4, 8, 8) + 1j * rs.randn(4, 8, 8)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr3, "name": "batch_4x8x8_c64"}))

    arr4 = (rs.randn(2, 3, 4, 6) + 1j * rs.randn(2, 3, 4, 6)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr4, "name": "nd_2x3x4x6_c128"}))

    arr5 = np.array([[3-4j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr5, "name": "single_1x1_c64"}))

    arr6 = (np.arange(7).reshape(1, 7) + 1j * (-np.arange(7).reshape(1, 7))).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr6, "name": "row_1x7_c128"}))

    arr7 = (np.linspace(-4, 4, 9).reshape(9, 1) + 1j * (-2.0) * np.ones((9, 1))).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr7, "name": "col_9x1_c64"}))

    arr8 = np.arange(36, dtype=np.float64).reshape(6, 6).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr8, "name": "real_only_6x6_c128"}))

    arr9 = (1j * np.arange(20).reshape(5, 4)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr9, "name": "imag_only_5x4_c64"}))

    arr10 = (rs.randn(3, 2, 5, 5) + 1j * rs.randn(3, 2, 5, 5)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr10, "name": "channels_3x2x5x5_c64"}))

    arr11 = (1e5 * (rs.randn(10, 10) + 1j * rs.randn(10, 10))).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr11, "name": "large_magnitude_10x10_c64"}))

    base = np.arange(49, dtype=np.float32).reshape(7, 7)
    arr12 = (base.T - 25 + 1j * (base[::-1, :] - 10)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr12, "name": "transposed_and_sliced_7x7_c64"}))

    return list_of_inputs

generated_inputs["tf.signal.fft2d"] = tf_signal_fft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.signal.fft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fft2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.signal.fft2d', generated_inputs['tf.signal.fft2d'], lib="tf", suffix=0)
