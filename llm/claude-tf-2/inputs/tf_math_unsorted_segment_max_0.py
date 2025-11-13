
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []
    
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.float32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "test1"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    num_segments = 3
    name = "test2"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([-5, -10, 15, -20, 25], dtype=np.float64)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    num_segments = 3
    name = "test3"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "test4"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([100, 200, 300, 400], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    num_segments = 1
    name = "test5"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, -1, 1, -1, 0], dtype=np.int32)
    num_segments = 2
    name = "test6"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([10, 20, 30, 40], dtype=np.uint8)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    num_segments = 2
    name = "test7"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    data = np.array([5, 15, 25, 35, 45], dtype=np.int16)
    segment_ids = np.array([2, 0

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max'], lib="tf", suffix=0)
