
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_linalg_linearoperatorzeros_inputs():
    list_of_inputs = []
    
    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "batch_shape": None,
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 3,
        "num_columns": 7,
        "batch_shape": None,
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "ZeroOperator"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 4,
        "num_columns": 4,
        "batch_shape": [2, 3],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "BatchZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 1,
        "num_columns": 1,
        "batch_shape": None,
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "SingleZero"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 10,
        "num_columns": 20,
        "batch_shape": [5],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "LargeZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 6,
        "num_columns": 6,
        "batch_shape": None,
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "ComplexZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 2,
        "num_columns": 15,
        "batch_shape": [3, 2],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "WideZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 8,
        

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorZeros' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorZeros'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorZeros', generated_inputs['tf.linalg.LinearOperatorZeros'], lib="tf", suffix=0)
