
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_experimental_numpy_rot90_inputs():
    list_of_inputs = []
    
    m = np.array([[1, 2], [3, 4]])
    k = 1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3], [4, 5, 6]])
    k = 2
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2], [3, 4], [5, 6]])
    k = 3
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[7, 8, 9], [10, 11, 12]])
    k = 4
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 1
    axes = (1, 2)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k = 2
    axes = (0, 2)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.random.rand(2, 3, 4, 5)
    k = 1
    axes = (1, 2)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1.5, 2.5], [3.5, 4.5]])
    k = 1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    k = -2
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.rot90'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.rot90', generated_inputs['tf.experimental.numpy.rot90'], lib="tf", suffix=0)
