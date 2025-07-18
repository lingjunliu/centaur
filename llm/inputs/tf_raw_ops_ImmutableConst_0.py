
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tempfile
import os
import time

# Assume generated_inputs is pre-initialized
# generated_inputs = {}

def create_memory_region_file(shape, dtype):
    """
    Creates a temporary file with zeroed data of the given shape and dtype,
    and returns its absolute path.
    The C++ implementation of the op expects a path to a file that it can mmap.
    This function ensures the file is created, fully written, and its path is returned.
    """
    # Using delete=False to ensure the file persists after the handle is closed,
    # so the TensorFlow op can read it. The testing framework is responsible
    # for cleaning up these files.
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        path = f.name  # This is an absolute path, which is what the C++ tests use.
        
        # Using np.zeros is the simplest way to get a correctly-sized buffer.
        data = np.zeros(shape, dtype=dtype)
        data_bytes = data.tobytes()
        
        f.write(data_bytes)
        
        # Ensure data is written to disk to avoid race conditions where TensorFlow
        # tries to read the file before the OS has finished writing it.
        f.flush()
        os.fsync(f.fileno())

    # Final sanity check to ensure the file size on disk matches expectations.
    # This is the most common failure point for this op.
    expected_size = data.nbytes
    actual_size = os.path.getsize(path)
    if expected_size != actual_size:
        # This case should ideally not be hit with the fsync above.
        raise RuntimeError(
            f"File size mismatch for {path}. "
            f"Expected: {expected_size}, Got: {actual_size}"
        )
        
    return path

def tf_raw_ops_ImmutableConst_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ImmutableConst function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D float32 tensor
    shape_1 = [2, 3]
    dtype_1 = np.float32
    input_dict_1 = {
        'dtype': dtype_1,
        'shape': shape_1,
        'memory_region_name': create_memory_region_file(shape_1, dtype_1),
        'name': 'const_float32_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D int64 tensor
    shape_2 = [10]
    dtype_2 = np.int64
    input_dict_2 = {
        'dtype': dtype_2,
        'shape': shape_2,
        'memory_region_name': create_memory_region_file(shape_2, dtype_2),
        'name': 'const_int64_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 0D (scalar) bool tensor
    shape_3 = []
    dtype_3 = np.bool_
    input_dict_3 = {
        'dtype': dtype_3,
        'shape': shape_3,
        'memory_region_name': create_memory_region_file(shape_3, dtype_3),
        'name': 'scalar_bool_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: High-dimensional float64 tensor
    shape_4 = [1, 2, 3, 4]
    dtype_4 = np.float64
    input_dict_4 = {
        'dtype': dtype_4,
        'shape': shape_4,
        'memory_region_name': create_memory_region_file(shape_4, dtype_4),
        'name': 'high_dim_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D complex64 tensor
    shape_5 = [4, 4]
    dtype_5 = np.complex64
    input_dict_5 = {
        'dtype': dtype_5,
        'shape': shape_5,
        'memory_region_name': create_memory_region_file(shape_5, dtype_5),
        'name': 'complex_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tensor with a zero dimension (empty tensor)
    shape_6 = [5, 0, 2]
    dtype_6 = np.uint8
    input_dict_6 = {
        'dtype': dtype_6,
        'shape': shape_6,
        'memory_region_name': create_memory_region_file(shape_6, dtype_6),
        'name': 'empty_tensor_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D int32 tensor
    shape_7 = [100]
    dtype_7 = np.int32
    input_dict_7 = {
        'dtype': dtype_7,
        'shape': shape_7,
        'memory_region_name': create_memory_region_file(shape_7, dtype_7),
        'name': 'another_int_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D int16 tensor
    shape_8 = [16, 16]
    dtype_8 = np.int16
    input_dict_8 = {
        'dtype': dtype_8,
        'shape': shape_8,
        'memory_region_name': create_memory_region_file(shape_8, dtype_8),
        'name': 'medium_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 2D float16 (half-precision) tensor
    shape_9 = [32, 16]
    dtype_9 = np.float16
    input_dict_9 = {
        'dtype': dtype_9,
        'shape': shape_9,
        'memory_region_name': create_memory_region_file(shape_9, dtype_9),
        'name': 'half_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 3D complex128 tensor
    shape_10 = [2, 2, 2]
    dtype_10 = np.complex128
    input_dict_10 = {
        'dtype': dtype_10,
        'shape': shape_10,
        'memory_region_name': create_memory_region_file(shape_10, dtype_10),
        'name': 'double_precision_complex_const'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ImmutableConst"] = tf_raw_ops_ImmutableConst_inputs()

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
