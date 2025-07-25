
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

# A custom variable class to satisfy the testing harness which expects a .size
# attribute, while also being a mutable tf.Variable for the API.
class PatchedVariable(tf.Variable):
    @property
    def size(self):
        return np.prod(self.shape.as_list()) if self.shape.as_list() else 0

def tf_raw_ops_apply_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, 1D
    list_of_inputs.append({
        'var': PatchedVariable(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        'alpha': np.array(0.1, dtype=np.float32),
        'delta': np.array([0.5, 0.4, 0.3], dtype=np.float32),
        'use_locking': False,
        'name': 'float32_1d_patched'
    })

    # Input 2: float64, 2D with locking
    list_of_inputs.append({
        'var': PatchedVariable(np.array([[10.0, -5.0], [8.0, -2.0]], dtype=np.float64)),
        'alpha': np.array(0.01, dtype=np.float64),
        'delta': np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
        'use_locking': True,
        'name': 'float64_2d_locked_patched'
    })

    # Input 3: int32, 1D with negative values
    list_of_inputs.append({
        'var': PatchedVariable(np.array([-100, 200, -300], dtype=np.int32)),
        'alpha': np.array(2, dtype=np.int32),
        'delta': np.array([10, -5, 15], dtype=np.int32),
        'use_locking': False,
        'name': 'int32_negatives_patched'
    })

    # Input 4: half (float16)
    list_of_inputs.append({
        'var': PatchedVariable(np.array([5.5, 6.6], dtype=np.float16)),
        'alpha': np.array(0.2, dtype=np.float16),
        'delta': np.array([1.1, 2.2], dtype=np.float16),
        'use_locking': False,
        'name': 'half_float16_patched'
    })

    # Input 5: bfloat16
    bfloat16_dtype = tf.bfloat16.as_numpy_dtype
    list_of_inputs.append({
        'var': PatchedVariable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=bfloat16_dtype)),
        'alpha': np.array(0.1, dtype=bfloat16_dtype),
        'delta': np.ones((2, 2), dtype=bfloat16_dtype),
        'use_locking': False,
        'name': 'bfloat16_patched'
    })

    # Input 6: Scalar var, alpha, and delta
    list_of_inputs.append({
        'var': PatchedVariable(np.array(100.0, dtype=np.float32)),
        'alpha': np.array(0.5, dtype=np.float32),
        'delta': np.array(10.0, dtype=np.float32),
        'use_locking': True,
        'name': 'scalar_all_patched'
    })

    # Input 7: complex64
    list_of_inputs.append({
        'var': PatchedVariable(np.array([1+2j, 3+4j], dtype=np.complex64)),
        'alpha': np.array(0.5+0j, dtype=np.complex64),
        'delta': np.array([0.2+0.1j, -0.4-0.3j], dtype=np.complex64),
        'use_locking': False,
        'name': 'complex64_patched'
    })

    # Input 8: 3D tensor
    list_of_inputs.append({
        'var': PatchedVariable(np.arange(8, dtype=np.float32).reshape((2, 2, 2))),
        'alpha': np.array(1.0, dtype=np.float32),
        'delta': np.ones((2, 2, 2), dtype=np.float32),
        'use_locking': True,
        'name': 'float32_3d_patched'
    })

    # Input 9: Zero-sized var and delta
    list_of_inputs.append({
        'var': PatchedVariable(np.array([], dtype=np.float32)),
        'alpha': np.array(0.1, dtype=np.float32),
        'delta': np.array([], dtype=np.float32),
        'use_locking': False,
        'name': 'zero_sized_patched'
    })
    
    # Input 10: int64
    list_of_inputs.append({
        'var': PatchedVariable(np.array([1000, 2000], dtype=np.int64)),
        'alpha': np.array(10, dtype=np.int64),
        'delta': np.array([5, 8], dtype=np.int64),
        'use_locking': False,
        'name': 'int64_patched'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyGradientDescent"] = tf_raw_ops_apply_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyGradientDescent'.")

check_valid('tf.raw_ops.ApplyGradientDescent', generated_inputs['tf.raw_ops.ApplyGradientDescent'], lib="tf", suffix=0)
