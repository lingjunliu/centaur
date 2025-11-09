
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []
    
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[0], [3], [7]], dtype=np.int64)
    values = np.array([5.5, 2.3, 8.1], dtype=np.float32)
    dense_shape = np.array([9], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[0, 0, 1], [1, 2, 0], [1, 2, 3]], dtype=np.int64)
    values = np.array([10, 20, 30], dtype=np.int32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[0, 1], [2, 3], [4, 0]], dtype=np.int64)
    values = np.array([-1.5, -2.7, -3.9], dtype=np.float64)
    dense_shape = np.array([5, 5], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[1, 1]], dtype=np.int64)
    values = np.array([42], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[0, 0], [0, 5], [3, 2], [5, 9], [9, 1]], dtype=np.int64)
    values = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    dense_shape = np.array([10, 10], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[0, 0, 0, 1], [1, 1, 1, 0]], dtype=np.int64)
    values = np.array([100, 200], dtype=np.int64)
    dense_shape = np.array([2, 2, 2, 2], dtype=np.int64)
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([[0, 0, 0], [2, 1, 3]], dtype=np.int64)
    values =

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.SparseTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.SparseTensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.SparseTensor', generated_inputs['tf.sparse.SparseTensor'], lib="tf", suffix=0)
