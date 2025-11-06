
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def index_copy_inputs():
    list_of_inputs = []
    
    input_t = torch.arange(5, dtype=torch.float32)
    dim = 0
    index_t = torch.tensor([1, 3], dtype=torch.int64)
    source_t = torch.tensor([-10.0, -30.0], dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    dim = 0
    index_t = torch.tensor([0, 2, 3], dtype=torch.int64)
    source_t = torch.tensor([[-1.0, -1.0, -1.0],
                             [-2.0, -2.0, -2.0],
                             [-3.0, -3.0, -3.0]], dtype=torch.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(15, dtype=torch.int32).reshape(3, 5)
    dim = 1
    index_t = torch.tensor([1, 4], dtype=torch.int64)
    source_t = torch.tensor([[-9, -8],
                             [-19, -18],
                             [-29, -28]], dtype=torch.int32)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3)
    dim = 1
    index_t = torch.tensor([0, 2], dtype=torch.int64)
    source_t = -torch.ones((2, 2, 3), dtype=torch.float16)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = (torch.arange(24).reshape(2, 3, 4) % 2 == 0)
    dim = -1
    index_t = torch.tensor([0, 2, 3], dtype=torch.int64)
    source_t = torch.ones((2, 3, 3), dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(3 * 2 * 4 * 5, dtype=torch.float32).reshape(3, 2, 4, 5)
    dim = 2
    index_t = torch.tensor([1, 3, 0], dtype=torch.int64)
    source_t = torch.full((3, 2, 3, 5), -1.25, dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(10, dtype=torch.int64).reshape(5, 2) - 5
    dim = 0
    index_t = torch.tensor([2, 2, 4], dtype=torch.int64)
    source_t = torch.tensor([[100, 101],
                             [200, 201],
                             [400, 401]], dtype=torch.int64)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    dim = 1
    index_t = torch.tensor([], dtype=torch.int64)
    source_t = torch.empty((2, 0, 4), dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(4 * 5 * 6, dtype=torch.float64).reshape(4, 5, 6)
    dim = -2
    index_t = torch.tensor([0, 1, 3, 4], dtype=torch.int64)
    source_t = torch.full((4, 4, 6), 7.5, dtype=torch.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.arange(2 * 2 * 3 * 4 * 5, dtype=torch.int16).reshape(2, 2, 3, 4, 5)
    dim = 3
    index_t = torch.tensor([1, 3], dtype=torch.int64)
    source_t = torch.full((2, 2, 3, 2, 5), -7, dtype=torch.int16)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.tensor([True, False, True], dtype=torch.bool)
    dim = 0
    index_t = torch.tensor([2], dtype=torch.int64)
    source_t = torch.tensor([False], dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    input_t = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]], dtype=torch.float32)
    dim = 1
    index_t = torch.tensor([2, 0], dtype=torch.int64)
    source_t = torch.tensor([[30.0, 10.0],
                             [60.0, 40.0],
                             [90.0, 70.0]], dtype=torch.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "dim": dim, "index": index_t.numpy(), "source": source_t.numpy()}))
    
    return list_of_inputs

generated_inputs["torch.index_copy"] = index_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.index_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_copy'.")


check_valid('torch.index_copy', generated_inputs['torch.index_copy'], lib="torch", suffix=0)
