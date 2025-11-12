
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []
    
    val = np.array([1, 2, 3, 4, 5])
    dtype = np.float32
    copy_flag = True
    ndmin = 0
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([[1, -2, 3], [-4, 5, -6]])
    dtype = np.int32
    copy_flag = False
    ndmin = 2
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = np.float64
    copy_flag = True
    ndmin = 3
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array(42)
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([0.0, 1.5, 2.7, 0.0])
    dtype = np.bool_
    copy_flag = True
    ndmin = 0
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([1+2j, 3+4j, 5+6j])
    dtype = np.complex128
    copy_flag = False
    ndmin = 1
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([])
    dtype = np.float32
    copy_flag = True
    ndmin = 1
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.random.randn(10, 10)
    dtype = np.float32
    copy_flag = True
    ndmin = 2
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([0, 128, 255])
    dtype = np.uint8
    copy_flag = False
    ndmin = 0
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy_flag,
        "ndmin": ndmin
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    val = np.array([[1.1, 2.2], [3.3, 4.4]])
    dtype = np.int16
    copy_flag = True

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.array' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array'], lib="tf", suffix=0)
