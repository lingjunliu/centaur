
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Imag_inputs():
    list_of_inputs = []

    arr1 = np.array([-2.25+4.75j, 3.25+5.75j, -0.0-7.125j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr1, "Tout": np.float32, "name": "imag_vec_c64_f32"}))

    arr2 = np.array([1+2j, -3-4.5j, 0+0j, 6-1e-3j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr2, "Tout": np.float64, "name": "imag_vec_c128_f64"}))

    arr3 = np.array([[-1+0.5j, 2-3j], [4+0j, -5-6j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr3, "Tout": np.float32, "name": "imag_mat_c64_f32"}))

    arr4 = np.array([[[1+1j, 2+2j], [3+3j, 4-4j]], [[-1-1j, -2+2j], [0+0j, 5-0.25j]]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr4, "Tout": np.float64, "name": "imag_3d_c128_f64"}))

    arr5 = np.array(3-7j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr5, "Tout": np.float32, "name": "imag_scalar_c64_f32"}))

    arr6 = np.array([], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr6, "Tout": np.float32, "name": "imag_empty1d_c64_f32"}))

    arr7 = np.empty((2, 0), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr7, "Tout": np.float64, "name": "imag_2x0_c128_f64"}))

    arr8 = np.array([np.nan + 1j, 2 + np.inf*1j, -np.inf + (-np.inf)*1j, np.nan + np.nan*1j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr8, "Tout": np.float64, "name": "imag_specials_c128_f64"}))

    arr9 = np.zeros((2, 1, 1, 3), dtype=np.complex64)
    arr9[0, 0, 0, :] = np.array([1+10j, -2-20j, 0+0j], dtype=np.complex64)
    arr9[1, 0, 0, :] = np.array([3-30j, -4+40j, 5-50j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr9, "Tout": np.float32, "name": "imag_4d_c64_f32"}))

    arr10 = np.array([1+0j, -2+0j, 3+0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr10, "Tout": np.float64, "name": "imag_zeroimag_c128_f64"}))

    base = np.array([0+1j, 1+2j, 2+3j, 3+4j, 4+5j, 5+6j], dtype=np.complex64)
    arr11 = base[::-2]
    list_of_inputs.append(copy.deepcopy({"input": arr11, "Tout": np.float32, "name": "imag_view_c64_f32"}))

    arr12 = np.array([1e-30+1e-30j, -1e30-1e30j, 3.141592653589793+2.718281828459045j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr12, "Tout": np.float64, "name": "imag_mixed_mags_c128_f64"}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Imag"] = tf_raw_ops_Imag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Imag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Imag', generated_inputs['tf.raw_ops.Imag'], lib="tf", suffix=0)
