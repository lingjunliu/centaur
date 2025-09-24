
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import tempfile
import os

# This set will store the paths of the created temporary files.
# It's important that these files are not deleted before the TensorFlow op uses them.
# In a real application, a cleanup mechanism would be needed.
_created_files_for_immutable_const = set()

def _create_tensor_file_for_immutable_const(shape, dtype):
    """
    Creates a temporary file containing raw bytes for a tensor of a given
    shape and dtype. Returns the path to the file.
    """
    # The op might fail on zero-sized tensors/files, so we skip them.
    if np.prod(shape) == 0 and len(shape) > 0:
        return None
        
    # Using np.zeros is fine as the content doesn't matter, only the size.
    arr = np.zeros(shape, dtype=dtype)
    tensor_bytes = arr.tobytes()

    # Create a named temporary file that is not deleted automatically on close.
    # This ensures the file exists when the TensorFlow op needs to read it.
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = f.name
            f.write(tensor_bytes)
        
        # Add to a global set to track them, though we don't clean them up here.
        _created_files_for_immutable_const.add(file_path)
        return file_path
    except Exception:
        # If file creation fails for any reason, we can't create a valid input.
        return None

def tf_raw_ops_immutable_const_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ImmutableConst function.
    """
    list_of_inputs = []

    # Define a list of test cases with different dtypes and shapes.
    # The case with a zero-sized dimension has been removed as it can be problematic.
    test_cases = [
        # (dtype, shape, name)
        (np.float32, [2, 3], 'const_float_2x3'),
        (np.int32, [10], 'const_int_vector'),
        (np.int64, [], 'const_scalar_int64'),
        (np.float64, [2, 2, 3, 4], 'const_high_rank_float64'),
        (np.bool_, [5, 5], 'const_bool_matrix'),
        (np.complex64, [4, 2], 'const_complex64_tensor'),
        (np.float16, [100], 'const_float16_vector'),
        (np.int16, [128, 128], 'const_medium_int16_matrix'),
        (np.complex128, [3, 3], 'const_complex128_matrix'),
        (np.uint32, [8, 8, 8], 'const_uint32_cube'),
    ]

    for dtype, shape, name in test_cases:
        # For each case, create a backing file.
        file_path = _create_tensor_file_for_immutable_const(shape, dtype)
        
        # Only proceed if the file was created successfully.
        if file_path:
            input_dict = {
                'dtype': dtype,
                'shape': shape,
                'memory_region_name': file_path,
                'name': name
            }
            list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ImmutableConst"] = tf_raw_ops_immutable_const_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ImmutableConst' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ImmutableConst'.")

check_valid('tf.raw_ops.ImmutableConst', generated_inputs['tf.raw_ops.ImmutableConst'], lib="tf", suffix=0)
