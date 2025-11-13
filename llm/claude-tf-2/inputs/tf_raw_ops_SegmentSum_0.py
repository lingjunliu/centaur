
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_sum_inputs():
    list_of_inputs = []
    
    data = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    name = "segment_sum_1"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    name = "segment_sum_2"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[-1, -2], [3, 4], [-5, 6]], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    name = "segment_sum_3"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    segment_ids = np.array([0, 0, 0], dtype=np.int32)
    name = "segment_sum_4"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    segment_ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    name = "segment_sum_5"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    name = "segment_sum_6"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.int16)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    name = "segment_sum_7"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([10, -5, 15, -20, 25], dtype=np.int8)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    name = "segment_sum_8"
    input_dict = {"name": name, "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([7, 8, 9, 10], dtype=np.uint16

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentSum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SegmentSum', generated_inputs['tf.raw_ops.SegmentSum'], lib="tf", suffix=0)
