
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_restorev2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RestoreV2 function.
    """
    list_of_inputs = []

    # Input 1: Basic case, 2 tensors, full restore
    input_dict_1 = {
        'prefix': np.array(b'/tmp/model/ckpt-1', dtype=object),
        'tensor_names': np.array([b'var/W', b'var/b'], dtype=object),
        'shape_and_slices': np.array([b'', b''], dtype=object),
        'dtypes': [tf.float32, tf.float32],
        'name': 'restore_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Single tensor, full restore, no name
    input_dict_2 = {
        'prefix': np.array(b'./my_checkpoint', dtype=object),
        'tensor_names': np.array([b'conv1/kernel'], dtype=object),
        'shape_and_slices': np.array([b''], dtype=object),
        'dtypes': [tf.float32]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Multiple tensors (4), mixed dtypes, full restore
    input_dict_3 = {
        'prefix': np.array(b'gs://bucket/path/to/model.ckpt', dtype=object),
        'tensor_names': np.array([b'weights', b'biases', b'step', b'is_training'], dtype=object),
        'shape_and_slices': np.array([b'', b'', b'', b''], dtype=object),
        'dtypes': [tf.float64, tf.float32, tf.int64, tf.bool],
        'name': 'restore_mixed_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Sliced restore for one tensor, full for another
    input_dict_4 = {
        'prefix': np.array(b'/tmp/model/partitioned_ckpt', dtype=object),
        'tensor_names': np.array([b'embedding_matrix', b'global_step'], dtype=object),
        'shape_and_slices': np.array([b'0 1000', b''], dtype=object),
        'dtypes': [tf.float32, tf.int64],
        'name': 'restore_partial_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: All tensors sliced with different rank slice specs
    input_dict_5 = {
        'prefix': np.array(b'/tmp/model/sliced_model', dtype=object),
        'tensor_names': np.array([b'layer1/weights', b'layer1/biases'], dtype=object),
        'shape_and_slices': np.array([b'0 128 0 64', b'0 64'], dtype=object),
        'dtypes': [tf.float16, tf.float16],
        'name': 'restore_all_sliced'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using int8 and uint8
    input_dict_6 = {
        'prefix': np.array(b'/home/user/data/checkpoint', dtype=object),
        'tensor_names': np.array([b'quantized_weights', b'zero_point'], dtype=object),
        'shape_and_slices': np.array([b'', b''], dtype=object),
        'dtypes': [tf.int8, tf.uint8],
        'name': 'restore_quantized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex types
    input_dict_7 = {
        'prefix': np.array(b's3://my-bucket/models/run_1/ckpt', dtype=object),
        'tensor_names': np.array([b'fourier_coeffs_64', b'fourier_coeffs_128'], dtype=object),
        'shape_and_slices': np.array([b'', b''], dtype=object),
        'dtypes': [tf.complex64, tf.complex128],
        'name': 'restore_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Restore a scalar tensor (empty slice spec)
    input_dict_8 = {
        'prefix': np.array(b'./scalar_ckpt', dtype=object),
        'tensor_names': np.array([b'config/learning_rate'], dtype=object),
        'shape_and_slices': np.array([b''], dtype=object),
        'dtypes': [tf.float32],
        'name': 'restore_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: More complex slice spec for a 3D tensor
    input_dict_9 = {
        'prefix': np.array(b'/data/3d_model/model.ckpt', dtype=object),
        'tensor_names': np.array([b'conv3d/kernel'], dtype=object),
        'shape_and_slices': np.array([b'0 3 0 3 0 32'], dtype=object),
        'dtypes': [tf.float32],
        'name': 'restore_3d_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using bfloat16 and different prefix format
    input_dict_10 = {
        'prefix': np.array(b'file:///c:/users/admin/tf_models/ckpt_final', dtype=object),
        'tensor_names': np.array([b'tpu_var', b'cpu_var'], dtype=object),
        'shape_and_slices': np.array([b'', b''], dtype=object),
        'dtypes': [tf.bfloat16, tf.int32],
        'name': 'restore_bfloat16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.RestoreV2"] = tf_raw_ops_restorev2_inputs()

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
