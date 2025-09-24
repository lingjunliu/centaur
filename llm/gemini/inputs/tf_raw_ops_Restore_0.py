
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os
import shutil

def get_tf_raw_ops_restore_inputs():
    """
    Generates valid inputs for tf.raw_ops.Restore.
    This involves creating physical checkpoint files on disk for the op to read.
    """

    # Helper to create a V1 checkpoint, which is what tf.raw_ops.Restore expects.
    def _create_v1_checkpoint(path_prefix, tensor_name, tensor_value):
        # Use a new graph and session for each save to ensure statelessness.
        graph = tf.Graph()
        with graph.as_default():
            # The name of the tf.Variable becomes the tensor_name in the checkpoint.
            var = tf.Variable(initial_value=tensor_value, name=tensor_name)
            # Saver needs to know which variable to save under which name.
            saver = tf.compat.v1.train.Saver({tensor_name: var})
            with tf.compat.v1.Session() as sess:
                # Initialize the variable before saving.
                sess.run(tf.compat.v1.global_variables_initializer())
                # The save op returns the full path to the checkpoint prefix.
                saved_path = saver.save(sess, path_prefix, write_meta_graph=False)
        return saved_path

    # Use a local directory in the current working directory for the checkpoints.
    # This avoids potential issues with temporary directories being cleaned up
    # by the OS or test harness between input generation and execution.
    ckpt_dir = "tf_restore_op_checkpoints"
    
    # Always start with a clean directory.
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir)
    
    # Use absolute paths to be robust against CWD changes.
    abs_ckpt_dir = os.path.abspath(ckpt_dir)

    test_cases = [
        # (name_suffix, numpy_value, preferred_shard)
        ("f32", np.array([1.0, 2.0], dtype=np.float32), -1),
        ("i32", np.array([[1, 2], [3, 4]], dtype=np.int32), 0),
        ("f64", np.array([3.14], dtype=np.float64), -1),
        ("b", np.array(True, dtype=np.bool_), 1),
        ("c64", np.array([1+2j], dtype=np.complex64), -1),
        ("i64", np.array([2**40], dtype=np.int64), -1),
        ("u8", np.array([0, 255], dtype=np.uint8), -1),
        ("f16", np.array([0.5, -0.5], dtype=np.float16), -1),
        ("s", np.array([b"abc", b"def"]), -1),
        ("i16", np.array([-100, 100], dtype=np.int16), -1),
    ]

    list_of_inputs = []
    for i, (name_suffix, value, shard) in enumerate(test_cases):
        tensor_name = f"var_{name_suffix}"
        path_prefix = os.path.join(abs_ckpt_dir, f"model_{i}.ckpt")
        
        try:
            # Create the actual checkpoint files on disk.
            saved_path = _create_v1_checkpoint(path_prefix, tensor_name, value)

            # TF's 'string' type corresponds to multiple numpy types.
            dt_val = value.dtype
            if dt_val.type in (np.bytes_, np.object_, np.string_):
                dt_val = np.string_

            # Construct the input dictionary for the op.
            input_dict = {
                'file_pattern': np.array([saved_path], dtype=object),
                'tensor_name': np.array([tensor_name], dtype=object),
                'dt': dt_val,
                'preferred_shard': shard,
                'name': f'restore_{tensor_name}'
            }
            list_of_inputs.append(copy.deepcopy(input_dict))

        except Exception:
            # If a specific checkpoint fails to be created, skip it.
            continue
            
    return list_of_inputs

generated_inputs["tf.raw_ops.Restore"] = get_tf_raw_ops_restore_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Restore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Restore'.")

check_valid('tf.raw_ops.Restore', generated_inputs['tf.raw_ops.Restore'], lib="tf", suffix=0)
