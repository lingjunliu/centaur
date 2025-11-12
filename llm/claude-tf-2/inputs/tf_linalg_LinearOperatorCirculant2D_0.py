
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_linalg_LinearOperatorCirculant2D_inputs():
    list_of_inputs = []
    
    spectrum = np.array([[1.0+0.0j, 2.0+0.0j, 3.0+0.0j],
                         [4.0+0.0j, 5.0+0.0j, 6.0+0.0j]], dtype=np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": None,
        "is_self_adjoint": None,
        "is_positive_definite": None,
        "is_square": True,
        "name": "LinearOperatorCirculant2D"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    spectrum = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "SelfAdjointCirculant"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    spectrum = np.array([[[1.0+1.0j, 2.0-1.0j],
                          [3.0+2.0j, 4.0-2.0j]],
                         [[5.0+0.5j, 6.0-0.5j],
                          [7.0+1.5j, 8.0-1.5j]]], dtype=np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "BatchedCirculant"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    spectrum = np.array([[1.0+0.0j, 2.0+1.0j],
                         [3.0-1.0j, 4.0+0.0j]], dtype=np.complex128)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex128,
        "is_non_singular": None,
        "is_self_adjoint": None,
        "is_positive_definite": None,
        "is_square": True,
        "name": "Complex128Circulant"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    spectrum = np.array([[2.0+1.0j, 3.0-1.0j, 3.0+1.0j],
                         [4.0-2.0j, 5.0+0.0j, 4.0+2.0j]], dtype=np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.float32,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "Float32Output"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    spectrum = np.array([[1.0+0.0j, 0.5+0.0j],
                         [0.5+0.0j, 2.0+0.0j]], dtype=

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorCirculant2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorCirculant2D', generated_inputs['tf.linalg.LinearOperatorCirculant2D'], lib="tf", suffix=0)
