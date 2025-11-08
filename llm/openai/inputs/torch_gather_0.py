
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def torch_gather_inputs():
    list_of_inputs = []
    
    # Input 1
    input_t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dim = 1
    index_t = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": False,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    dim = 0
    index_t = torch.tensor([[1, 0, 1]], dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": True,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_t = torch.randn(3, 4, 5, dtype=torch.float64)
    dim = 2
    index_t = torch.randint(0, 5, (3, 4, 2), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": False,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_t = torch.randn(3, 4, 5, dtype=torch.float16)
    dim = 1
    index_t = torch.randint(0, 4, (3, 2, 5), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": True,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_t = torch.tensor([10.0, -20.0, 30.0, -40.0, 50.0], dtype=torch.float32)
    dim = 0
    index_t = torch.tensor([4, 0, 2], dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": False,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_t = torch.randint(-100, 100, (2, 3, 4, 5), dtype=torch.int32)
    dim = -1
    index_t = torch.randint(0, 5, (2, 3, 4, 2), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": True,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_t = torch.tensor([[[[True, False], [False, True]],
                              [[True, True], [False, False]]],
                             [[[False, True], [True, False]],
                              [[False, False], [True, True]]]], dtype=torch.bool)
    dim = 2
    index_t = torch.randint(0, 2, (2, 2, 1, 2), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": False,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_t = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32)
    dim = -3
    index_t = torch.randint(0, 3, (1, 2, 2, 4, 5), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": True,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 (empty index)
    input_t = torch.randn(3, 4, dtype=torch.float64)
    dim = 1
    index_t = torch.zeros((3, 0), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": False,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_t = torch.randint(0, 256, (2, 1, 3), dtype=torch.uint8)
    dim = 2
    index_t = torch.randint(0, 3, (2, 1, 4), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": True,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_t = torch.randn(5, 2, 2, dtype=torch.float32)
    dim = 0
    index_t = torch.randint(0, 5, (2, 2, 2), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": False,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_t = torch.randn(4, 5, 6, dtype=torch.float32)
    dim = -2
    index_t = torch.randint(0, 5, (4, 3, 6), dtype=torch.long)
    out_t = torch.empty(index_t.shape, dtype=input_t.dtype)
    input_dict = {
        "input": input_t.numpy(),
        "dim": dim,
        "index": index_t.numpy(),
        "sparse_grad": True,
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.gather"] = torch_gather_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gather'.")


check_valid('torch.gather', generated_inputs['torch.gather'], lib="torch", suffix=0)
