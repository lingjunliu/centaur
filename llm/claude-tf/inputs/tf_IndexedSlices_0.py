
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_IndexedSlices_inputs():
    list_of_inputs = []
    
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    dense_shape = np.array([5, 3], dtype=np.int64)
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    indices = np.array([3], dtype=np.int32)
    dense_shape = np.array([10, 3], dtype=np.int64)
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices = np.array([1, 5], dtype=np.int32)
    dense_shape = np.array([10, 2, 2], dtype=np.int64)
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    dense_shape = np.array([6, 2], dtype=np.int64)
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    indices = np.array([1, 3], dtype=np.int64)
    dense_shape = None
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.random.randn(3, 2, 2, 4).astype(np.float32)
    indices = np.array([0, 5, 9], dtype=np.int32)
    dense_shape = np.array([10, 2, 2, 4], dtype=np.int64)
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    indices = np.array([2, 7], dtype=np.int32)
    dense_shape = np.array([10, 3], dtype=np.int64)
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    values = np.array([[0.0, 0.0]], dtype=np.float32)
    indices = np.array([0], dtype=np.int

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices'], lib="tf", suffix=0)
