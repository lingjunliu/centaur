
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DynamicPartition_inputs():
    list_of_inputs = []

    # Input 1: Scalar partition, 1D data
    list_of_inputs.append({
        "name": "scalar_partition",
        "data": np.array([10, 20], dtype=np.float32),
        "partitions": np.array(1, dtype=np.int32),
        "num_partitions": 2
    })

    # Input 2: Vector partition, 1D integer data
    list_of_inputs.append({
        "name": "vector_partition_int",
        "data": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        "partitions": np.array([0, 0, 1, 1, 0], dtype=np.int32),
        "num_partitions": 2
    })

    # Input 3: Vector partition, multi-dimensional float data
    list_of_inputs.append({
        "name": "vector_partition_multi_dim_float",
        "data": np.random.randn(3, 4, 5).astype(np.float32),
        "partitions": np.array([0, 2, 1], dtype=np.int32),
        "num_partitions": 3
    })

    # Input 4: 2D partition, 3D float data
    list_of_inputs.append({
        "name": "2d_partition_float",
        "data": np.random.randn(2, 2, 3).astype(np.float32),
        "partitions": np.array([[0, 1], [2, 0]], dtype=np.int32),
        "num_partitions": 3
    })

    # Input 5: Large number of partitions, float64 data
    list_of_inputs.append({
        "name": "large_partitions",
        "data": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),
        "partitions": np.array([5, 4, 3, 2, 1, 0], dtype=np.int32),
        "num_partitions": 6
    })

    # Input 6: Unused partition indices
    list_of_inputs.append({
        "name": "unused_partitions",
        "data": np.array([1, 2, 3], dtype=np.int32),
        "partitions": np.array([0, 0, 0], dtype=np.int32),
        "num_partitions": 3
    })

    # Input 7: Boolean data type
    list_of_inputs.append({
        "name": "bool_data",
        "data": np.array([True, False, True, True], dtype=np.bool_),
        "partitions": np.array([1, 0, 1, 0], dtype=np.int32),
        "num_partitions": 2
    })

    # Input 8: Complex64 data type
    list_of_inputs.append({
        "name": "complex_data",
        "data": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        "partitions": np.array([0, 1, 0], dtype=np.int32),
        "num_partitions": 2
    })

    # Input 9: 3D partition with 4D data
    partitions_3d = np.zeros((2, 2, 2), dtype=np.int32)
    partitions_3d[0, 1, 1] = 1
    partitions_3d[1, 0, 1] = 1
    list_of_inputs.append({
        "name": "3d_partition_4d_data",
        "data": np.random.randint(0, 100, size=(2, 2, 2, 5), dtype=np.int32),
        "partitions": partitions_3d,
        "num_partitions": 2
    })

    # Input 10: Single partition
    list_of_inputs.append({
        "name": "single_partition",
        "data": np.array([10, 20, 30, 40], dtype=np.int32),
        "partitions": np.array([0, 0, 0, 0], dtype=np.int32),
        "num_partitions": 1
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.DynamicPartition"] = tf_raw_ops_DynamicPartition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DynamicPartition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DynamicPartition'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DynamicPartition', generated_inputs['tf.raw_ops.DynamicPartition'], lib="tf", suffix=0)
