
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import io

def jit_load_inputs():
    list_of_inputs = []

    # Create a dummy ScriptModule for testing
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module = torch.jit.script(DummyModule())

    # Save the module to a buffer
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)

    # Input 1: Load from buffer with default parameters
    input_dict = {
        'f': buffer,
        'map_location': None,
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Reset buffer for subsequent uses
    buffer.seek(0)

    # Input 2: Load from buffer, map to CPU
    input_dict = {
        'f': buffer,
        'map_location': 'cpu',
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Reset buffer for subsequent uses
    buffer.seek(0)

    # Input 3: Load from buffer, map to CUDA (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dict = {
        'f': buffer,
        'map_location': device,
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Reset buffer for subsequent uses
    buffer.seek(0)

    # Input 4: Load from buffer with extra files
    extra_files = {'test.txt': 'This is a test file.'}
    input_dict = {
        'f': buffer,
        'map_location': None,
        '_extra_files': extra_files,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Reset buffer for subsequent uses
    buffer.seek(0)
    
    # Input 5: Load from buffer, restore shapes
    input_dict = {
        'f': buffer,
        'map_location': None,
        '_extra_files': None,
        '_restore_shapes': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Reset buffer for subsequent uses
    buffer.seek(0)

    # Input 6: Load from string filename
    temp_file = "temp_scriptmodule.pt"
    torch.jit.save(module, temp_file)
    input_dict = {
        'f': temp_file,
        'map_location': None,
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = jit_load_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('load', generated_inputs)
