
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restore_slice_inputs():
    list_of_inputs = []

    # Input 1
    file_pattern = np.array("checkpoint_file", dtype=np.string_)
    tensor_name = np.array("my_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 10 20, slice: 0 0 10 20", dtype=np.string_)
    dt = tf.float32.as_numpy_dtype
    preferred_shard = -1
    name = "restore_slice_op_1"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    file_pattern = np.array("another_checkpoint", dtype=np.string_)
    tensor_name = np.array("another_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 5 5 5, slice: 0 0 0 5 5 5", dtype=np.string_)
    dt = tf.int32.as_numpy_dtype
    preferred_shard = 0
    name = "restore_slice_op_2"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    file_pattern = np.array("third_checkpoint", dtype=np.string_)
    tensor_name = np.array("third_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 256, slice: 0 0 256", dtype=np.string_)
    dt = tf.float64.as_numpy_dtype
    preferred_shard = 1
    name = "restore_slice_op_3"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    file_pattern = np.array("fourth_checkpoint", dtype=np.string_)
    tensor_name = np.array("fourth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 128 64, slice: 0 0 128 64", dtype=np.string_)
    dt = tf.complex64.as_numpy_dtype
    preferred_shard = -1
    name = "restore_slice_op_4"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    file_pattern = np.array("fifth_checkpoint", dtype=np.string_)
    tensor_name = np.array("fifth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 32 32 32, slice: 0 0 0 32 32 32", dtype=np.string_)
    dt = tf.bool.as_numpy_dtype
    preferred_shard = 0
    name = "restore_slice_op_5"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    file_pattern = np.array("sixth_checkpoint", dtype=np.string_)
    tensor_name = np.array("sixth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 1024, slice: 0 0 1024", dtype=np.string_)
    dt = tf.int64.as_numpy_dtype
    preferred_shard = 1
    name = "restore_slice_op_6"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    file_pattern = np.array("seventh_checkpoint", dtype=np.string_)
    tensor_name = np.array("seventh_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 8 8 8 8, slice: 0 0 0 0 8 8 8 8", dtype=np.string_)
    dt = tf.bfloat16.as_numpy_dtype
    preferred_shard = -1
    name = "restore_slice_op_7"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    file_pattern = np.array("eighth_checkpoint", dtype=np.string_)
    tensor_name = np.array("eighth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 16 16, slice: 0 0 16 16", dtype=np.string_)
    dt = tf.qint8.as_numpy_dtype
    preferred_shard = 0
    name = "restore_slice_op_8"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    file_pattern = np.array("ninth_checkpoint", dtype=np.string_)
    tensor_name = np.array("ninth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 4 4 4 4 4, slice: 0 0 0 0 0 4 4 4 4 4", dtype=np.string_)
    dt = tf.quint8.as_numpy_dtype
    preferred_shard = 1
    name = "restore_slice_op_9"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    file_pattern = np.array("tenth_checkpoint", dtype=np.string_)
    tensor_name = np.array("tenth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 2 2 2 2 2 2, slice: 0 0 0 0 0 0 2 2 2 2 2 2", dtype=np.string_)
    dt = tf.float32.as_numpy_dtype # Replace tf.resource with a supported dtype
    preferred_shard = -1
    name = "restore_slice_op_10"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Fixing the dtype for file_pattern, tensor_name and shape_and_slice
    file_pattern = np.array("eleventh_checkpoint", dtype=np.string_)
    tensor_name = np.array("eleventh_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 64 64, slice: 0 0 64 64", dtype=np.string_)
    dt = tf.float32.as_numpy_dtype
    preferred_shard = 0
    name = "restore_slice_op_11"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - Different Slice
    file_pattern = np.array("twelfth_checkpoint", dtype=np.string_)
    tensor_name = np.array("twelfth_tensor", dtype=np.string_)
    shape_and_slice = np.array("shape: 128 128, slice: 10 20 30 40", dtype=np.string_) # Invalid Slice, but testing different values
    dt = tf.float32.as_numpy_dtype
    preferred_shard = -1
    name = "restore_slice_op_12"
    input_dict = {
        "file_pattern": file_pattern,
        "tensor_name": tensor_name,
        "shape_and_slice": shape_and_slice,
        "dt": dt,
        "preferred_shard": preferred_shard,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["tf.raw_ops.RestoreSlice"] = tf_raw_ops_restore_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RestoreSlice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreSlice'.")

check_valid('tf.raw_ops.RestoreSlice', generated_inputs['tf.raw_ops.RestoreSlice'], lib="tf", suffix=0)
