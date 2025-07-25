
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_SparseAccumulatorApplyGradient_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseAccumulatorApplyGradient function.
    NOTE: This operation is not compatible with eager execution and will raise a
    RuntimeError if called directly. This is an inherent property of the op, which
    is designed for TensorFlow's graph mode. The inputs provided are syntactically
    and semantically valid for the op's documented behavior in a graph context.
    """
    list_of_inputs = []

    def create_handle(i):
        # A placeholder string tensor for the handle, as required by the signature.
        return np.array([f'handle_placeholder_{i}'], dtype=np.object_)

    # Input 1: Basic float32 case, updating slices of a 2D tensor
    list_of_inputs.append({
        'handle': create_handle(1),
        'local_step': np.array(10, dtype=np.int64),
        'gradient_indices': np.array([1, 3], dtype=np.int64),
        'gradient_values': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'gradient_shape': np.array([5, 2], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_float32'
    })

    # Input 2: Basic int32 case, updating elements of a 1D tensor
    list_of_inputs.append({
        'handle': create_handle(2),
        'local_step': np.array(20, dtype=np.int64),
        'gradient_indices': np.array([0, 1, 4], dtype=np.int64),
        'gradient_values': np.array([-5, 10, -15], dtype=np.int32),
        'gradient_shape': np.array([10], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_int32'
    })

    # Input 3: float64, updating a slice of a 3D tensor
    list_of_inputs.append({
        'handle': create_handle(3),
        'local_step': np.array(30, dtype=np.int64),
        'gradient_indices': np.array([0], dtype=np.int64),
        'gradient_values': np.array([[[1.1, 1.2], [2.1, 2.2]]], dtype=np.float64),
        'gradient_shape': np.array([3, 2, 2], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_float64_3d'
    })

    # Input 4: complex128, 1D tensor
    list_of_inputs.append({
        'handle': create_handle(4),
        'local_step': np.array(40, dtype=np.int64),
        'gradient_indices': np.array([4, 9], dtype=np.int64),
        'gradient_values': np.array([1+2j, -3-4j], dtype=np.complex128),
        'gradient_shape': np.array([10], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_complex128'
    })

    # Input 5: Empty gradient update
    list_of_inputs.append({
        'handle': create_handle(5),
        'local_step': np.array(50, dtype=np.int64),
        'gradient_indices': np.array([], dtype=np.int64),
        'gradient_values': np.empty((0, 5, 5), dtype=np.float32),
        'gradient_shape': np.array([10, 5, 5], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_empty'
    })

    # Input 6: has_known_shape = False
    list_of_inputs.append({
        'handle': create_handle(6),
        'local_step': np.array(60, dtype=np.int64),
        'gradient_indices': np.array([0], dtype=np.int64),
        'gradient_values': np.array([1.0], dtype=np.float32),
        'gradient_shape': np.array([-1], dtype=np.int64),
        'has_known_shape': False,
        'name': 'apply_grad_unknown_shape'
    })

    # Input 7: uint16 type
    list_of_inputs.append({
        'handle': create_handle(7),
        'local_step': np.array(70, dtype=np.int64),
        'gradient_indices': np.array([0, 100], dtype=np.int64),
        'gradient_values': np.array([65535, 1], dtype=np.uint16),
        'gradient_shape': np.array([101], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_uint16'
    })

    # Input 8: int8 type, 2D slices
    list_of_inputs.append({
        'handle': create_handle(8),
        'local_step': np.array(80, dtype=np.int64),
        'gradient_indices': np.array([1], dtype=np.int64),
        'gradient_values': np.array([[-127, 0, 127]], dtype=np.int8),
        'gradient_shape': np.array([5, 3], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_int8'
    })

    # Input 9: uint32 with max value
    list_of_inputs.append({
        'handle': create_handle(9),
        'local_step': np.array(90, dtype=np.int64),
        'gradient_indices': np.array([0], dtype=np.int64),
        'gradient_values': np.array([np.iinfo(np.uint32).max], dtype=np.uint32),
        'gradient_shape': np.array([1], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_uint32'
    })

    # Input 10: Max local_step value
    list_of_inputs.append({
        'handle': create_handle(10),
        'local_step': np.array(9223372036854775807, dtype=np.int64),
        'gradient_indices': np.array([0], dtype=np.int64),
        'gradient_values': np.array([1.0], dtype=np.float32),
        'gradient_shape': np.array([1], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_large_step'
    })

    # Input 11: bfloat16 placeholder
    list_of_inputs.append({
        'handle': create_handle(11),
        'local_step': np.array(110, dtype=np.int64),
        'gradient_indices': np.array([0], dtype=np.int64),
        'gradient_values': np.array([1.0], dtype=np.float32), # Use float32 as numpy doesn't have bfloat16
        'gradient_shape': np.array([1], dtype=np.int64),
        'has_known_shape': True,
        'name': 'apply_grad_bfloat16'
    })

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.SparseAccumulatorApplyGradient"] = get_tf_raw_ops_SparseAccumulatorApplyGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseAccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseAccumulatorApplyGradient'.")

check_valid('tf.raw_ops.SparseAccumulatorApplyGradient', generated_inputs['tf.raw_ops.SparseAccumulatorApplyGradient'], lib="tf", suffix=0)
