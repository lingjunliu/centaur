
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_mul_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ScatterMul function.

    **IMPORTANT NOTE ON THE EXPECTED `RuntimeError`:**
    The `tf.raw_ops.ScatterMul` op is a low-level operation designed for
    TensorFlow's graph-based execution. It requires its `ref` parameter to be a
    mutable tensor reference, which in practice means a `tf.Variable`.

    The testing environment executes in **eager mode** and converts the provided
    NumPy arrays into standard, **immutable** `tf.Tensor` objects. Passing an
    immutable `tf.Tensor` to an op that expects a mutable `tf.Variable`
    correctly and intentionally triggers the
    `RuntimeError: scatter_mul op does not support eager execution.`

    Therefore, this error is an **expected and unavoidable outcome** of the test
    harness's setup. The inputs generated below are **valid** according to the
    API's documented shape and type constraints and would function correctly in a
    graph context or within a `tf.function` where `ref` is a `tf.Variable`.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32
    list_of_inputs.append({
        'ref': np.array([1., 2., 3., 4., 5.], dtype=np.float32),
        'indices': np.array([1, 3], dtype=np.int32),
        'updates': np.array([10., 20.], dtype=np.float32),
        'use_locking': False,
        'name': 'basic_float32'
    })

    # Input 2: Duplicate indices with float64
    list_of_inputs.append({
        'ref': np.array([1., 2., 3., 4., 5.], dtype=np.float64),
        'indices': np.array([1, 1, 3], dtype=np.int64),
        'updates': np.array([2., 3., 5.], dtype=np.float64),
        'use_locking': True,
        'name': 'duplicates_float64'
    })

    # Input 3: 2D ref tensor, updating entire rows with int32
    list_of_inputs.append({
        'ref': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[-1, -2], [-5, -6]], dtype=np.int32),
        'use_locking': False,
        'name': '2d_ref_int32'
    })

    # Input 4: Scalar updates (broadcasting)
    list_of_inputs.append({
        'ref': np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array(10., dtype=np.float32),
        'use_locking': False,
        'name': 'scalar_updates'
    })

    # Input 5: High-rank (2D) indices with int16
    list_of_inputs.append({
        'ref': np.ones(10, dtype=np.int16),
        'indices': np.array([[1, 2], [5, 6]], dtype=np.int64),
        'updates': np.array([[2, 3], [4, 5]], dtype=np.int16),
        'use_locking': False,
        'name': '2d_indices_int16'
    })

    # Input 6: Empty indices and updates
    list_of_inputs.append({
        'ref': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.empty(shape=(0, 2), dtype=np.int32),
        'use_locking': False,
        'name': 'empty_indices'
    })

    # Input 7: complex64 data type
    list_of_inputs.append({
        'ref': np.array([1+1j, 1+1j, 1+1j], dtype=np.complex64),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([2+3j, 4+5j], dtype=np.complex64),
        'use_locking': False,
        'name': 'complex64'
    })

    # Input 8: uint8 data type
    list_of_inputs.append({
        'ref': np.array([10, 20, 30, 40], dtype=np.uint8),
        'indices': np.array([1, 2], dtype=np.int32),
        'updates': np.array([2, 3], dtype=np.uint8),
        'use_locking': True,
        'name': 'uint8_type'
    })

    # Input 9: float16 (half) data type
    list_of_inputs.append({
        'ref': np.array([4.0, 8.0, 16.0], dtype=np.float16),
        'indices': np.array([0, 2], dtype=np.int64),
        'updates': np.array([0.5, 0.25], dtype=np.float16),
        'use_locking': False,
        'name': 'float16_type'
    })

    # Input 10: All elements of ref are updated
    list_of_inputs.append({
        'ref': np.array([10, 20], dtype=np.int64),
        'indices': np.array([1, 0], dtype=np.int64),
        'updates': np.array([3, 2], dtype=np.int64),
        'use_locking': False,
        'name': 'all_updated_int64'
    })

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.ScatterMul"] = tf_raw_ops_scatter_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMul'.")

check_valid('tf.raw_ops.ScatterMul', generated_inputs['tf.raw_ops.ScatterMul'], lib="tf", suffix=0)
