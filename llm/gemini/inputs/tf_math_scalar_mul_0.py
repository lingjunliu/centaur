
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_scalar_mul_inputs():
    list_of_inputs = []

    # Input 1: Float32 scalar and 1D float32 array
    list_of_inputs.append({
        "scalar": np.array(2.5, dtype=np.float32),
        "x": np.array([1.0, -2.0, 3.0], dtype=np.float32),
        "name": "scale_1d_float"
    })

    # Input 2: Int32 negative scalar and 2D int32 array
    list_of_inputs.append({
        "scalar": np.array(-5, dtype=np.int32),
        "x": np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32),
        "name": "scale_2d_int"
    })

    # Input 3: Float64 zero scalar and 3D float64 array
    list_of_inputs.append({
        "scalar": np.array(0.0, dtype=np.float64),
        "x": np.arange(24, dtype=np.float64).reshape((2, 3, 4)),
        "name": "scale_3d_zero"
    })

    # Input 4: Complex scalar and 1D complex array
    list_of_inputs.append({
        "scalar": np.array(1.5 + 2.0j, dtype=np.complex64),
        "x": np.array([1.0 - 1.0j, 2.0 + 3.0j], dtype=np.complex64),
        "name": "scale_complex"
    })

    # Input 5: Float32 scalar and 0D float32 scalar tensor
    list_of_inputs.append({
        "scalar": np.array(-1.2, dtype=np.float32),
        "x": np.array(4.5, dtype=np.float32),
        "name": "scale_0d"
    })

    # Input 6: Large float32 scalar and 4D float32 array
    list_of_inputs.append({
        "scalar": np.array(100.0, dtype=np.float32),
        "x": np.ones((2, 2, 2, 2), dtype=np.float32),
        "name": "scale_4d_large"
    })

    # Input 7: Float16 scalar and 2D float16 array
    list_of_inputs.append({
        "scalar": np.array(0.5, dtype=np.float16),
        "x": np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float16),
        "name": "scale_float16"
    })

    # Input 8: Int64 scalar and 1D int64 array
    list_of_inputs.append({
        "scalar": np.array(10, dtype=np.int64),
        "x": np.array([100, 200, 300], dtype=np.int64),
        "name": "scale_int64"
    })

    # Input 9: Small float64 scalar and 5D float64 array
    list_of_inputs.append({
        "scalar": np.array(1e-5, dtype=np.float64),
        "x": np.ones((1, 2, 1, 2, 1), dtype=np.float64),
        "name": "scale_5d_small"
    })

    # Input 10: Negative float32 scalar and large 3D float32 array
    list_of_inputs.append({
        "scalar": np.array(-0.1, dtype=np.float32),
        "x": np.random.randn(5, 5, 5).astype(np.float32),
        "name": "scale_random_3d"
    })

    return list_of_inputs

generated_inputs["tf.math.scalar_mul"] = tf_math_scalar_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.scalar_mul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.scalar_mul'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.scalar_mul', generated_inputs['tf.math.scalar_mul'], lib="tf", suffix=0)
