
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSegmentMean_inputs():
    list_of_inputs = []
    
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_mean_1"
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    sparse_gradient = True
    name = "sparse_segment_mean_2"
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_mean_3"
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    sparse_gradient = True
    name = "sparse_segment_mean_4"
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int64)
    segment_ids = np.array([0, 1, 1], dtype=np.int64)
    sparse_gradient = False
    name = "sparse_segment_mean_5"
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[2.0], [4.0], [6.

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSegmentMean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentMean'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSegmentMean', generated_inputs['tf.raw_ops.SparseSegmentMean'], lib="tf", suffix=0)
