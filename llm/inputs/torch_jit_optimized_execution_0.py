
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

@torch.jit.script
def _inner_callable_for_optimized_execution(a, b):
    return a + b

def optimized_execution_inputs():
    list_of_inputs = []

    # The error "returns a function, but the input does not have inner values"
    # indicates the test harness requires a special key to provide the callable
    # and arguments to be executed within the context manager.
    # This attempt uses the key `_inner_input_`, a plausible name for such a mechanism,
    # and avoids using `copy.deepcopy` to prevent the `PickleError` that can occur
    # with `torch.jit.ScriptFunction` objects. This combination of a specific key name
    # and correct handling of the callable has not been tried before and addresses
    # the sequence of errors encountered.

    # Input 1: enabled=True, float32 tensors
    list_of_inputs.append({
        'enabled': np.bool_(True),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.randn(2, 3, dtype=torch.float32).numpy(),
                torch.randn(2, 3, dtype=torch.float32).numpy()
            )
        }
    })

    # Input 2: enabled=False, float32 tensors
    list_of_inputs.append({
        'enabled': np.bool_(False),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.ones(5, dtype=torch.float32).numpy(),
                torch.ones(5, dtype=torch.float32).numpy()
            )
        }
    })

    # Input 3: enabled=True, int32 tensors
    list_of_inputs.append({
        'enabled': np.bool_(True),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).numpy(),
                torch.tensor([[5, 6], [7, 8]], dtype=torch.int32).numpy()
            )
        }
    })

    # Input 4: enabled=False, float64 tensors
    list_of_inputs.append({
        'enabled': np.bool_(False),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.zeros((1, 4), dtype=torch.float64).numpy(),
                torch.ones((1, 4), dtype=torch.float64).numpy()
            )
        }
    })

    # Input 5: enabled=True, 0-dim tensors (scalars)
    list_of_inputs.append({
        'enabled': np.bool_(True),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.tensor(5.0).numpy(),
                torch.tensor(-3.0).numpy()
            )
        }
    })

    # Input 6: enabled=False, 3D tensors
    list_of_inputs.append({
        'enabled': np.bool_(False),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.randn(3, 1, 2).numpy(),
                torch.randn(3, 1, 2).numpy()
            )
        }
    })

    # Input 7: enabled=True, int64 tensors with negative values
    list_of_inputs.append({
        'enabled': np.bool_(True),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.tensor([-10, -20, -30], dtype=torch.int64).numpy(),
                torch.tensor([10, 20, 30], dtype=torch.int64).numpy()
            )
        }
    })

    # Input 8: enabled=False, different shapes for broadcasting
    list_of_inputs.append({
        'enabled': np.bool_(False),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.randn(3, 1).numpy(),
                torch.randn(1, 3).numpy()
            )
        }
    })

    # Input 9: enabled=True, scientific notation scalars
    list_of_inputs.append({
        'enabled': np.bool_(True),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.tensor(1.23e4, dtype=torch.float32).numpy(),
                torch.tensor(4.56e-2, dtype=torch.float32).numpy()
            )
        }
    })

    # Input 10: enabled=False, integer tensors with large range
    list_of_inputs.append({
        'enabled': np.bool_(False),
        '_inner_input_': {
            'callable': _inner_callable_for_optimized_execution,
            'args': (
                torch.randint(-100, 100, (2, 2, 2), dtype=torch.int32).numpy(),
                torch.randint(-100, 100, (2, 2, 2), dtype=torch.int32).numpy()
            )
        }
    })

    return list_of_inputs

generated_inputs["torch.jit.optimized_execution"] = optimized_execution_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.optimized_execution' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.optimized_execution'.")

check_valid('torch.jit.optimized_execution', generated_inputs['torch.jit.optimized_execution'], lib="torch", suffix=0)
