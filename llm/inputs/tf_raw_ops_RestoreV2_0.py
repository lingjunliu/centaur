
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf
import os
import tempfile

def tf_raw_ops_restore_v2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RestoreV2 function.
    This function creates temporary checkpoint files to ensure the restore
    operation can find them and execute successfully.
    """
    list_of_inputs = []
    try:
        # A temporary directory to store checkpoints.
        temp_dir = tempfile.mkdtemp()
    except Exception:
        # Fallback if mkdtemp fails
        temp_dir = "./tf_raw_ops_restore_v2_temp"
        os.makedirs(temp_dir, exist_ok=True)


    # Input 1: Restore a single, full, float32 tensor.
    prefix_1 = os.path.join(temp_dir, "ckpt_1")
    tensor_names_1 = ["v1"]
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_1, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_1, dtype=tf.string),
        shape_and_slices=tf.constant([""]),
        tensors=[tf.constant([1.0, 2.0], dtype=tf.float32)]
    )
    input_dict_1 = {
        'prefix': np.array([prefix_1], dtype=object),
        'tensor_names': np.array(tensor_names_1, dtype=object),
        'shape_and_slices': np.array([""], dtype=object),
        'dtypes': [1],  # tf.float32
        'name': 'restore_single_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Restore multiple full tensors with mixed dtypes.
    prefix_2 = os.path.join(temp_dir, "ckpt_2")
    tensor_names_2 = ["weights", "biases", "step"]
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_2, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_2, dtype=tf.string),
        shape_and_slices=tf.constant(["", "", ""]),
        tensors=[
            tf.constant([[1.0], [2.0]], dtype=tf.float64),
            tf.constant([0.1], dtype=tf.float64),
            tf.constant(100, dtype=tf.int64)
        ]
    )
    input_dict_2 = {
        'prefix': np.array([prefix_2], dtype=object),
        'tensor_names': np.array(tensor_names_2, dtype=object),
        'shape_and_slices': np.array(["", "", ""], dtype=object),
        'dtypes': [2, 2, 9],  # tf.float64, tf.float64, tf.int64
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Restore a slice from a 2D tensor.
    prefix_3 = os.path.join(temp_dir, "ckpt_3")
    tensor_names_3 = ["large_matrix"]
    large_matrix = tf.reshape(tf.range(100, dtype=tf.int32), (10, 10))
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_3, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_3, dtype=tf.string),
        shape_and_slices=tf.constant([""]),
        tensors=[large_matrix]
    )
    input_dict_3 = {
        'prefix': np.array([prefix_3], dtype=object),
        'tensor_names': np.array(tensor_names_3, dtype=object),
        'shape_and_slices': np.array(["5 1 0 5"], dtype=object),
        'dtypes': [3],  # tf.int32
        'name': 'restore_2d_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Restore a mix of a full tensor and a sliced tensor.
    prefix_4 = os.path.join(temp_dir, "ckpt_4")
    tensor_names_4 = ["config_flags", "image_data"]
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_4, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_4, dtype=tf.string),
        shape_and_slices=tf.constant(["", ""]),
        tensors=[
            tf.constant([True, False, True], dtype=tf.bool),
            tf.zeros((10, 5, 3), dtype=tf.uint8)
        ]
    )
    input_dict_4 = {
        'prefix': np.array([prefix_4], dtype=object),
        'tensor_names': np.array(tensor_names_4, dtype=object),
        'shape_and_slices': np.array(["", "1 8 2 2 0 2"], dtype=object),
        'dtypes': [10, 4],  # tf.bool, tf.uint8
        'name': 'restore_mixed_slice_full'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Restore tensors with complex and string types.
    prefix_5 = os.path.join(temp_dir, "ckpt_5")
    tensor_names_5 = ["complex_coeffs", "metadata"]
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_5, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_5, dtype=tf.string),
        shape_and_slices=tf.constant(["", ""]),
        tensors=[
            tf.constant([1+2j, -3-4j], dtype=tf.complex64),
            tf.constant(["model_v1", "trained_on_gpu"], dtype=tf.string)
        ]
    )
    input_dict_5 = {
        'prefix': np.array([prefix_5], dtype=object),
        'tensor_names': np.array(tensor_names_5, dtype=object),
        'shape_and_slices': np.array(["", ""], dtype=object),
        'dtypes': [8, 7],  # tf.complex64, tf.string
        'name': 'restore_complex_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Restore slice from 1D tensor with float16.
    prefix_6 = os.path.join(temp_dir, "ckpt_6")
    tensor_names_6 = ["one_d_array"]
    one_d_array = tf.range(50, dtype=tf.float16)
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_6, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_6, dtype=tf.string),
        shape_and_slices=tf.constant([""]),
        tensors=[one_d_array]
    )
    input_dict_6 = {
        'prefix': np.array([prefix_6], dtype=object),
        'tensor_names': np.array(tensor_names_6, dtype=object),
        'shape_and_slices': np.array(["5 9"], dtype=object),
        'dtypes': [19],  # tf.float16
        'name': 'restore_1d_slice_fp16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty slice spec (full restore) for an int8 tensor.
    prefix_7 = os.path.join(temp_dir, "ckpt_7")
    tensor_names_7 = ["partitioned_var_slice"]
    tf.raw_ops.SaveV2(
        prefix=tf.constant(prefix_7, dtype=tf.string),
        tensor_names=tf.constant(tensor_names_7, dtype=tf.string),
        shape_and_slices=tf.constant([""]),
        tensors=[tf.constant([5, 6, 7, 8], dtype=tf.int8)]
    )
    input_dict_7 = {
        'prefix': np.array([prefix_7], dtype=object),
        'tensor_names': np.array(tensor_names_7, dtype=object),
        'shape_and_slices': np.array([""], dtype=object),
        'dtypes': [6],  # tf.int8
        'name': 'restore_empty_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Restore slice that covers the full tensor
    prefix_8 = os.path.join(temp_dir, "ckpt_8")
    tensor_names_8 = ["full_slice"]
    tensor_8 = tf.constant([[1,2],[3,4]], dtype=tf.int16)
    tf.raw_ops.SaveV2(prefix=tf.constant(prefix_8, dtype=tf.string), tensor_names=tf.constant(tensor_names_8, dtype=tf.string), shape_and_slices=tf.constant([""]), tensors=[tensor_8])
    input_dict_8 = {'prefix': np.array([prefix_8], dtype=object), 'tensor_names': np.array(tensor_names_8, dtype=object), 'shape_and_slices': np.array(["0 1 0 1"], dtype=object), 'dtypes': [5], 'name': 'restore_full_slice'}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Restore tensor with a long name
    prefix_9 = os.path.join(temp_dir, "ckpt_9")
    tensor_names_9 = ["a/very/long/and/nested/variable/name"]
    tf.raw_ops.SaveV2(prefix=tf.constant(prefix_9, dtype=tf.string), tensor_names=tf.constant(tensor_names_9, dtype=tf.string), shape_and_slices=tf.constant([""]), tensors=[tf.constant(1.0)])
    input_dict_9 = {'prefix': np.array([prefix_9], dtype=object), 'tensor_names': np.array(tensor_names_9, dtype=object), 'shape_and_slices': np.array([""], dtype=object), 'dtypes': [1], 'name': 'restore_long_name'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Restore a scalar tensor
    prefix_10 = os.path.join(temp_dir, "ckpt_10")
    tensor_names_10 = ["scalar_value"]
    tf.raw_ops.SaveV2(prefix=tf.constant(prefix_10, dtype=tf.string), tensor_names=tf.constant(tensor_names_10, dtype=tf.string), shape_and_slices=tf.constant([""]), tensors=[tf.constant(3.14, dtype=tf.float32)])
    input_dict_10 = {'prefix': np.array([prefix_10], dtype=object), 'tensor_names': np.array(tensor_names_10, dtype=object), 'shape_and_slices': np.array([""], dtype=object), 'dtypes': [1], 'name': 'restore_scalar'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.RestoreV2"] = tf_raw_ops_restore_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RestoreV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreV2'.")

check_valid('tf.raw_ops.RestoreV2', generated_inputs['tf.raw_ops.RestoreV2'], lib="tf", suffix=0)
