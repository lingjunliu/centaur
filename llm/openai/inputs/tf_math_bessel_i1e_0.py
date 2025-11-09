
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_bessel_i1e_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = "vec_float32_basic"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0.1, -2.5, 5.0], [10.0, -7.5, 0.0]], dtype=np.float64)
    name = "matrix_float64_mixed"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(2.5, dtype=np.float32)
    name = "scalar_float32_positive"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0],
                  [4.0, -4.0, 2.0, -2.0, 0.5]], dtype=np.float16)
    name = "matrix_float16_varied"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    base = np.arange(24, dtype=np.float64).reshape(4, 6) - 12.0
    x = base[:, ::-2]
    name = "noncontiguous_slice_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.asfortranarray(np.array([[0.01, 0.1, 1.0],
                                    [2.0, 5.0, 10.0],
                                    [-1.0, -0.1, -0.01]], dtype=np.float32))
    name = "fortran_order_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.RandomState(42).uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    name = "tensor3d_float64_random"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-50.0, -20.0, 0.0, 20.0, 50.0, 100.0], dtype=np.float64)
    name = "large_magnitude_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((5,), dtype=np.float16)
    name = "zeros_vector_float16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.full((2, 2, 2), 1e-6, dtype=np.float32)
    name = "small_values_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float64)
    name = "nan_inf_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.transpose(np.arange(12, dtype=np.float32).reshape(3, 4))
    name = "transposed_noncontiguous_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float64)
    name = "empty_array_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.bessel_i1e"] = tf_math_bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.bessel_i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i1e'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.bessel_i1e', generated_inputs['tf.math.bessel_i1e'], lib="tf", suffix=0)
