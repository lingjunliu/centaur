
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_strided_slice_inputs():
    list_of_inputs = []
    
    input_ = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
    begin = [1, 0, 0]
    end = [2, 1, 3]
    strides = [1, 1, 1]
    var = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
    input_dict = {"input_": input_, "begin": begin, "end": end, "strides": strides, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "var": var, "name": "slice1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_ = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
    begin = [1, -1, 0]
    end = [2, -3, 3]
    strides = [1, -1, 1]
    var = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
    input_dict = {"input_": input_, "begin": begin, "end": end, "strides": strides, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "var": var, "name": "slice2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_ = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    begin = [0, 2]
    end = [2, 4]
    strides = [1, 1]
    var = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    input_dict = {"input_": input_, "begin": begin, "end": end, "strides": strides, "begin_mask": 1, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "var": var, "name": "slice3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_ = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    begin = [0]
    end = [8]
    strides = [2]
    var = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    input_dict = {"input_": input_, "begin": begin, "end": end, "strides": strides, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "var": var, "name": "slice4"}

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strided_slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strided_slice'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strided_slice', generated_inputs['tf.strided_slice'], lib="tf", suffix=0)
