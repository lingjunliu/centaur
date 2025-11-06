
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def index_add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy()
    dim = 0
    index = torch.tensor([0, 2, 4], dtype=torch.long).numpy()
    source = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64, dim=0
    input = torch.arange(12, dtype=torch.float64).reshape(4, 3).numpy()
    dim = 0
    index = torch.tensor([1, 3], dtype=torch.long).numpy()
    source = torch.tensor([[10.0, 10.0, 10.0],
                           [-5.0, -5.0, -5.0]], dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int64, dim=1
    input = torch.tensor([[1, 2, 3],
                          [4, 5, 6]], dtype=torch.int64).numpy()
    dim = 1
    index = torch.tensor([2, 0], dtype=torch.long).numpy()
    source = torch.tensor([[100, 200],
                           [300, 400]], dtype=torch.int64).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32, dim=-1, duplicates in index
    input = torch.zeros((2, 3, 4), dtype=torch.float32).numpy()
    dim = -1
    index = torch.tensor([3, 1, 1], dtype=torch.long).numpy()
    source = torch.tensor([[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]],
                           [[-1.0, -2.0, -3.0],
                            [-4.0, -5.0, -6.0],
                            [-7.0, -8.0, -9.0]]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32, dim=1
    input = torch.tensor([[[1, 2],
                           [3, 4],
                           [5, 6]],
                          [[7, 8],
                           [9, 10],
                           [11, 12]]], dtype=torch.int32).numpy()
    dim = 1
    index = torch.tensor([2, 1], dtype=torch.long).numpy()
    source = torch.tensor([[[10, 10],
                            [20, 20]],
                           [[30, 30],
                            [40, 40]]], dtype=torch.int32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float16, dim=0
    input = torch.ones((3, 2, 2, 2), dtype=torch.float16).numpy()
    dim = 0
    index = torch.tensor([0, 2], dtype=torch.long).numpy()
    source = torch.tensor([[[[0.1, 0.2],
                             [0.3, 0.4]],
                            [[0.5, 0.6],
                             [0.7, 0.8]]],
                           [[[1.1, 1.2],
                             [1.3, 1.4]],
                            [[1.5, 1.6],
                             [1.7, 1.8]]]], dtype=torch.float16).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32
    input = torch.tensor([10, 20, 30, 40], dtype=torch.int32).numpy()
    dim = 0
    index = torch.tensor([3, 0], dtype=torch.long).numpy()
    source = torch.tensor([5, -3], dtype=torch.int32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32, dim=-1, repeated indices
    input = torch.zeros((3, 5), dtype=torch.float32).numpy()
    dim = -1
    index = torch.tensor([4, 0, 2, 2], dtype=torch.long).numpy()
    source = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float64, dim=2
    input = torch.arange(0, 2*1*3*2*2, dtype=torch.float64).reshape(2, 1, 3, 2, 2).numpy()
    dim = 2
    index = torch.tensor([0, 1, 2], dtype=torch.long).numpy()
    source = input.copy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int64, dim=0
    input = torch.arange(60, dtype=torch.int64).reshape(3, 4, 5).numpy()
    dim = 0
    index = torch.tensor([2, 0], dtype=torch.long).numpy()
    source = torch.arange(40, dtype=torch.int64).reshape(2, 4, 5).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float32, dim=0, single index
    input = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=torch.float32).numpy()
    dim = 0
    index = torch.tensor([1], dtype=torch.long).numpy()
    source = torch.tensor([[10.0, 20.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D float32, dim=3
    input = torch.zeros((2, 2, 2, 4), dtype=torch.float32).numpy()
    dim = 3
    index = torch.tensor([3, 1], dtype=torch.long).numpy()
    source = torch.tensor([[[[1.0, 2.0],
                             [3.0, 4.0]],
                            [[5.0, 6.0],
                             [7.0, 8.0]]],
                           [[[9.0, 10.0],
                             [11.0, 12.0]],
                            [[13.0, 14.0],
                             [15.0, 16.0]]]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "source": source}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.index_add"] = index_add_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.index_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_add'.")


check_valid('torch.index_add', generated_inputs['torch.index_add'], lib="torch", suffix=0)
