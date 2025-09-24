
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os
import tempfile

def get_tf_raw_ops_restore_slice_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RestoreSlice function.
    This function first creates a temporary checkpoint file to ensure the
    restore operation can find the necessary files.
    """
    
    # --- Checkpoint Setup ---
    # Create a temporary directory and a checkpoint file with various tensors.
    # This setup is necessary for the RestoreSlice op to execute without a NotFoundError.
    # Note: tf.raw_ops.Save has limited dtype support, so bfloat16 and uint32 are excluded.
    temp_dir = tempfile.mkdtemp()
    file_prefix = os.path.join(temp_dir, "model.ckpt")
    
    with tf.device('CPU:0'):
        tensor_names_list = [
            "weights_float", "biases_int", "conv_kernel_double",
            "mask_bool", "scalar_lr", "feature_names_str",
            "complex_weights", "high_rank_fp16", "quantized_int8",
            "embedding_table_fp32"
        ]
        
        data_list = [
            tf.constant(np.arange(20, dtype=np.float32)),
            tf.constant(np.arange(200, dtype=np.int32).reshape(20, 10)),
            tf.constant(np.random.rand(5, 5, 64, 32).astype(np.float64)),
            tf.constant((np.arange(10000) % 2 == 0).reshape(100, 100)),
            tf.constant(0.001, dtype=tf.float32),
            tf.constant([b"feat1", b"feat2", b"feat3"], dtype=tf.string),
            tf.constant(np.random.rand(16, 16) + 1j * np.random.rand(16, 16), dtype=np.complex64),
            tf.constant(np.random.rand(2, 3, 4, 5, 6).astype(np.float16)),
            tf.constant(np.arange(1000, dtype=np.int8)),
            tf.constant(np.random.rand(1000, 768).astype(np.float32)),
        ]

        tensor_names_tensor = tf.constant(tensor_names_list, dtype=tf.string)
        tf.raw_ops.Save(filename=tf.constant(file_prefix), tensor_names=tensor_names_tensor, data=data_list)
    # --- End Checkpoint Setup ---

    list_of_inputs = []
    
    # Input 1: Basic case, 1D float tensor
    input_dict_1 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("weights_float", dtype=object),
        'shape_and_slice': np.array("20 0,10", dtype=object),
        'dt': np.float32,
        'preferred_shard': -1,
        'name': 'restore_slice_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int32 tensor, specific shard
    input_dict_2 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("biases_int", dtype=object),
        'shape_and_slice': np.array("20 10 5,10:-", dtype=object),
        'dt': np.int32,
        'preferred_shard': 0,
        'name': 'restore_slice_int32_shard0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 4D float64 tensor, complex slice
    input_dict_3 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("conv_kernel_double", dtype=object),
        'shape_and_slice': np.array("5 5 64 32 0,2:0,2:16,16:-", dtype=object),
        'dt': np.float64,
        'preferred_shard': -1,
        'name': 'restore_conv_kernel_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Boolean tensor
    input_dict_4 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("mask_bool", dtype=object),
        'shape_and_slice': np.array("100 100 -:50,50", dtype=object),
        'dt': np.bool_,
        'preferred_shard': -1,
        'name': 'restore_mask'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar tensor (full restore)
    input_dict_5 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("scalar_lr", dtype=object),
        'shape_and_slice': np.array("", dtype=object),
        'dt': np.float32,
        'preferred_shard': -1,
        'name': 'restore_scalar_lr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: String tensor
    input_dict_6 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("feature_names_str", dtype=object),
        'shape_and_slice': np.array("3 1,2", dtype=object),
        'dt': tf.string,
        'preferred_shard': -1,
        'name': 'restore_feature_names'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex64 tensor
    input_dict_7 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("complex_weights", dtype=object),
        'shape_and_slice': np.array("16 16 0,8:8,8", dtype=object),
        'dt': np.complex64,
        'preferred_shard': -1,
        'name': 'restore_complex_weights'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High rank tensor (5D) with float16
    input_dict_8 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("high_rank_fp16", dtype=object),
        'shape_and_slice': np.array("2 3 4 5 6 0,1:1,1:2,1:3,1:4,1", dtype=object),
        'dt': np.float16,
        'preferred_shard': 0,
        'name': 'restore_high_rank'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Int8 tensor, large dimension
    input_dict_9 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("quantized_int8", dtype=object),
        'shape_and_slice': np.array("1000 500,500", dtype=object),
        'dt': np.int8,
        'preferred_shard': -1,
        'name': 'restore_quantized_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Full restore of a 2D tensor using slice spec "-:-"
    input_dict_10 = {
        'file_pattern': np.array(file_prefix, dtype=object),
        'tensor_name': np.array("embedding_table_fp32", dtype=object),
        'shape_and_slice': np.array("1000 768 -:-", dtype=object),
        'dt': np.float32,
        'preferred_shard': -1,
        'name': 'restore_full_embedding_table'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.RestoreSlice"] = get_tf_raw_ops_restore_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RestoreSlice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RestoreSlice'.")

check_valid('tf.raw_ops.RestoreSlice', generated_inputs['tf.raw_ops.RestoreSlice'], lib="tf", suffix=0)
