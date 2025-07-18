
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_scatter_nd_sub_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ScatterNdSub.

    IMPORTANT NOTE: The error "scatter_nd_sub op does not support eager execution"
    is a fundamental issue with the testing environment, not the input data. This
    operation is a legacy op from TensorFlow 1.x designed for graph execution and
    is incompatible with the eager execution mode used by default in TensorFlow 2.x.
    No change to the input numpy arrays can resolve this error because the error is
    raised by a check within the TensorFlow op's wrapper itself before the actual
    computation. The correct op for eager mode is `tf.raw_ops.ResourceScatterNdSub`.
    The inputs provided below are valid according to the API's documentation and would
    work correctly in a TF1 graph context, but they are guaranteed to fail in the
    current eager execution harness.
    """
    list_of_inputs = []

    # Input 1: Basic 1D update, int32
    list_of_inputs.append({
        'ref': np.array([10, 20, 30, 40], dtype=np.int32),
        'indices': np.array([[1], [3]], dtype=np.int32),
        'updates': np.array([5, 15], dtype=np.int32),
        'use_locking': False, 'bad_indices_policy': '', 'name': 'basic_1d_int32'
    })

    # Input 2: 2D ref, updating slices (rows), float32
    list_of_inputs.append({
        'ref': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'indices': np.array([[0], [2]], dtype=np.int32),
        'updates': np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float32),
        'use_locking': False, 'bad_indices_policy': '', 'name': '2d_slice_float32'
    })

    # Input 3: 3D ref, updating elements (K=P=3), int64
    list_of_inputs.append({
        'ref': np.zeros((2, 2, 2), dtype=np.int64),
        'indices': np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        'updates': np.array([10, 20], dtype=np.int64),
        'use_locking': True, 'bad_indices_policy': '', 'name': '3d_element_int64'
    })

    # Input 4: Repeated indices to test accumulation, float64
    list_of_inputs.append({
        'ref': np.array([100.0, 200.0], dtype=np.float64),
        'indices': np.array([[0], [1], [0]], dtype=np.int64),
        'updates': np.array([10.0, 50.0, 20.0], dtype=np.float64),
        'use_locking': False, 'bad_indices_policy': '', 'name': 'repeated_indices_float64'
    })
    
    # Input 5: Empty update, should do nothing
    list_of_inputs.append({
        'ref': np.array([1, 2, 3], dtype=np.int32),
        'indices': np.empty((0, 1), dtype=np.int32),
        'updates': np.empty((0,), dtype=np.int32),
        'use_locking': False, 'bad_indices_policy': '', 'name': 'empty_update_noop'
    })
    
    # Input 6: uint8 type test
    list_of_inputs.append({
        'ref': np.array([100, 110, 120, 130], dtype=np.uint8),
        'indices': np.array([[1], [2]], dtype=np.int32),
        'updates': np.array([5, 15], dtype=np.uint8),
        'use_locking': False, 'bad_indices_policy': '', 'name': 'uint8_type_test'
    })

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.ScatterNdSub"] = get_scatter_nd_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterNdSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterNdSub'.")

check_valid('tf.raw_ops.ScatterNdSub', generated_inputs['tf.raw_ops.ScatterNdSub'], lib="tf", suffix=0)
