
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf
import os
import tempfile

def tf_raw_ops_restore_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Restore function.
    This function creates V1-style checkpoint files that the op can read,
    and ensures the op is called in an eager context, returning an EagerTensor.
    """
    list_of_inputs = []
    # Create a single temporary directory for all checkpoint files
    temp_dir = tempfile.mkdtemp()

    def create_checkpoint(file_path_prefix, tensor_name, tensor_value, sharded=False):
        """Helper to create a V1-style checkpoint file within an isolated graph."""
        graph = tf.Graph()
        with graph.as_default():
            var = tf.compat.v1.Variable(tensor_value, name=tensor_name)
            saver = tf.compat.v1.train.Saver(sharded=sharded)
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                saved_path = saver.save(sess, file_path_prefix, write_meta_graph=False)
        return saved_path

    def create_input_dict(file_pattern_str, tensor_name_str, dt_np, preferred_shard_int, name_str):
        """Helper to create the input dictionary with scalar tensors."""
        return {
            'file_pattern': np.array(file_pattern_str, dtype=object),
            'tensor_name': np.array(tensor_name_str, dtype=object),
            'dt': dt_np,
            'preferred_shard': preferred_shard_int,
            'name': name_str
        }

    # Input 1: Basic case with float32
    tensor_name_1 = "weights_float32"
    tensor_value_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    path_1 = create_checkpoint(os.path.join(temp_dir, "model1"), tensor_name_1, tensor_value_1)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_1, tensor_name_1, np.float32, -1, "RestoreFloat32")))

    # Input 2: Integer type
    tensor_name_2 = "biases_int32"
    tensor_value_2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    path_2 = create_checkpoint(os.path.join(temp_dir, "model2"), tensor_name_2, tensor_value_2)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_2, tensor_name_2, np.int32, 0, "RestoreInt32")))

    # Input 3: Double (float64) type
    tensor_name_3 = "high_precision_weights"
    tensor_value_3 = np.random.rand(5).astype(np.float64)
    path_3 = create_checkpoint(os.path.join(temp_dir, "model3"), tensor_name_3, tensor_value_3)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_3, tensor_name_3, np.float64, -1, "RestoreFloat64")))

    # Input 4: Half (float16) type
    tensor_name_4 = "half_precision_biases"
    tensor_value_4 = np.array([0.1, 0.2], dtype=np.float16)
    path_4 = create_checkpoint(os.path.join(temp_dir, "model4"), tensor_name_4, tensor_value_4)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_4, tensor_name_4, np.float16, 0, "RestoreFloat16")))

    # Input 5: Complex number type (complex64)
    tensor_name_5 = "fft_weights_complex64"
    tensor_value_5 = np.array([1+2j, 3+4j], dtype=np.complex64)
    path_5 = create_checkpoint(os.path.join(temp_dir, "model5"), tensor_name_5, tensor_value_5)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_5, tensor_name_5, np.complex64, -1, "RestoreComplex64")))
    
    # Input 6: Boolean type
    tensor_name_6 = "flags_bool"
    tensor_value_6 = np.array([True, False, True], dtype=np.bool_)
    path_6 = create_checkpoint(os.path.join(temp_dir, "model6"), tensor_name_6, tensor_value_6)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_6, tensor_name_6, np.bool_, 0, "RestoreBool")))

    # Input 7: 64-bit integer type
    tensor_name_7 = "global_step_int64"
    tensor_value_7 = np.array(100000, dtype=np.int64)
    path_7 = create_checkpoint(os.path.join(temp_dir, "model7"), tensor_name_7, tensor_value_7)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_7, tensor_name_7, np.int64, -1, "RestoreInt64")))
    
    # Input 8: Sharded checkpoint with wildcard
    shard_tensor_name = "sharded_variable"
    shard_tensor_value = np.random.rand(100, 100).astype(np.float32)
    path_prefix_8 = os.path.join(temp_dir, "sharded_model")
    create_checkpoint(path_prefix_8, shard_tensor_name, shard_tensor_value, sharded=True)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_prefix_8 + "*", shard_tensor_name, np.float32, 0, "RestoreSharded")))

    # Input 9: 8-bit integer for quantized models
    tensor_name_9 = "quantized_int8"
    tensor_value_9 = np.array([-128, 0, 127], dtype=np.int8)
    path_9 = create_checkpoint(os.path.join(temp_dir, "model9"), tensor_name_9, tensor_value_9)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_9, tensor_name_9, np.int8, 0, "RestoreInt8")))

    # Input 10: Unsigned 16-bit integer
    tensor_name_10 = "counters_uint16"
    tensor_value_10 = np.array([65535, 100, 0], dtype=np.uint16)
    path_10 = create_checkpoint(os.path.join(temp_dir, "model10"), tensor_name_10, tensor_value_10)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_10, tensor_name_10, np.uint16, -1, "RestoreUint16")))

    # Input 11: High-precision complex number (complex128)
    tensor_name_11 = "fft_weights_complex128"
    tensor_value_11 = np.array([1.5+2.5j, 3.5-4.5j], dtype=np.complex128)
    path_11 = create_checkpoint(os.path.join(temp_dir, "model11"), tensor_name_11, tensor_value_11)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_11, tensor_name_11, np.complex128, -1, "RestoreComplex128")))
    
    # Input 12: A scalar tensor
    tensor_name_12 = "scalar_float"
    tensor_value_12 = np.float32(3.14)
    path_12 = create_checkpoint(os.path.join(temp_dir, "model12"), tensor_name_12, tensor_value_12)
    list_of_inputs.append(copy.deepcopy(create_input_dict(path_12, tensor_name_12, np.float32, -1, "RestoreScalar")))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Restore"] = tf_raw_ops_restore_inputs()

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
