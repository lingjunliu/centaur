
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_padv2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.PadV2 operation.
    """
    list_of_inputs = []

    # Input 1: Basic 2D padding with integers
    list_of_inputs.append(
        {
            "name": "basic_2d_padding",
            "input": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "paddings": np.array([[1, 1], [1, 1]], dtype=np.int32),
            "constant_values": np.array(0, dtype=np.int32),
        }
    )

    # Input 2: Example from the documentation
    list_of_inputs.append(
        {
            "name": "doc_example",
            "input": np.array([[1, 1], [2, 2]], dtype=np.int32),
            "paddings": np.array([[1, 1], [2, 2]], dtype=np.int32),
            "constant_values": np.array(0, dtype=np.int32),
        }
    )

    # Input 3: 2D padding with float32 data and a negative constant value
    list_of_inputs.append(
        {
            "name": "float32_padding_negative_constant",
            "input": np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
            "paddings": np.array([[0, 2], [1, 3]], dtype=np.int32),
            "constant_values": np.array(-1.5, dtype=np.float32),
        }
    )

    # Input 4: 1D padding with int64 data
    list_of_inputs.append(
        {
            "name": "1d_padding_int64",
            "input": np.array([10, 20, 30], dtype=np.int64),
            "paddings": np.array([[3, 2]], dtype=np.int64),
            "constant_values": np.array(9, dtype=np.int64),
        }
    )

    # Input 5: 3D padding with float64 data and asymmetric padding
    list_of_inputs.append(
        {
            "name": "3d_padding_float64_asymmetric",
            "input": np.arange(8, dtype=np.float64).reshape((2, 2, 2)),
            "paddings": np.array([[1, 0], [0, 2], [3, 1]], dtype=np.int32),
            "constant_values": np.array(0.5, dtype=np.float64),
        }
    )

    # Input 6: No padding (paddings are all zero)
    list_of_inputs.append(
        {
            "name": "no_padding",
            "input": np.array([[10, 20], [30, 40]], dtype=np.int32),
            "paddings": np.array([[0, 0], [0, 0]], dtype=np.int32),
            "constant_values": np.array(99, dtype=np.int32),
        }
    )

    # Input 7: 4D padding
    list_of_inputs.append(
        {
            "name": "4d_padding",
            "input": np.ones((1, 2, 1, 3), dtype=np.float32),
            "paddings": np.array([[1, 1], [2, 0], [0, 3], [1, 1]], dtype=np.int32),
            "constant_values": np.array(3.14, dtype=np.float32),
        }
    )

    # Input 8: Padding only on one side of a dimension
    list_of_inputs.append(
        {
            "name": "one_sided_padding",
            "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            "paddings": np.array([[2, 0], [0, 3]], dtype=np.int32),
            "constant_values": np.array(-5, dtype=np.int32),
        }
    )

    # Input 9: Padding an empty tensor (shape [0])
    list_of_inputs.append(
        {
            "name": "pad_empty_1d_tensor",
            "input": np.array([], dtype=np.float32),
            "paddings": np.array([[2, 3]], dtype=np.int32),
            "constant_values": np.array(1.0, dtype=np.float32),
        }
    )

    # Input 10: Padding a tensor with an empty dimension (shape [1, 0, 2])
    list_of_inputs.append(
        {
            "name": "pad_empty_dimension",
            "input": np.zeros((1, 0, 2), dtype=np.int32),
            "paddings": np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32),
            "constant_values": np.array(7, dtype=np.int32),
        }
    )
    
    # Input 11: 5D padding
    list_of_inputs.append(
        {
            "name": "5d_padding",
            "input": np.arange(16, dtype=np.float32).reshape((1, 2, 2, 2, 2)),
            "paddings": np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int32),
            "constant_values": np.array(100.0, dtype=np.float32),
        }
    )

    return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.raw_ops.PadV2"] = get_tf_raw_ops_padv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PadV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PadV2'.")

check_valid('tf.raw_ops.PadV2', generated_inputs['tf.raw_ops.PadV2'], lib="tf", suffix=0)
