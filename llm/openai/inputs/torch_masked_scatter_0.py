
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def masked_scatter_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input = torch.tensor([1.0, -2.5, 3.3, 4.4, -5.5, 6.6], dtype=torch.float32).numpy()
    mask = torch.tensor([False, True, False, True, False, True], dtype=torch.bool).numpy()
    source = torch.tensor([-10.0, -20.0, -30.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 2: 2D int64
    input = torch.tensor([[1, 2, 3],
                          [4, 5, 6]], dtype=torch.int64).numpy()
    mask = torch.tensor([[True, False, True],
                         [True, False, True]], dtype=torch.bool).numpy()
    source = torch.tensor([-1, -2, 100, 200], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 3: 3D float64, source longer than needed
    input = torch.tensor([[[0.0, 1.0, 2.0],
                           [3.0, 4.0, 5.0]],
                          [[6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0]]], dtype=torch.float64).numpy()
    mask = torch.tensor([[[False, False, True],
                          [True, False, False]],
                         [[False, False, True],
                          [False, True, True]]], dtype=torch.bool).numpy()
    source = torch.tensor([-1.1, 2.2, -3.3, 4.4, 5.5, -6.6, 7.7], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 4: 4D uint8
    input = torch.arange(24, dtype=torch.uint8).view(1, 2, 3, 4).numpy()
    mflat = torch.zeros(24, dtype=torch.bool)
    mflat[torch.tensor([0, 5, 11, 12, 17, 23])] = True
    mask = mflat.view(1, 2, 3, 4).numpy()
    source = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 5: 2D float16, all False mask, empty source
    input = torch.tensor([[1.5, -2.0, 3.25],
                          [4.75, -5.5, 6.125],
                          [0.0, 1.0, -1.0]], dtype=torch.float16).numpy()
    mask = torch.zeros((3, 3), dtype=torch.bool).numpy()
    source = torch.tensor([], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 6: 2D int32, all True mask
    input = torch.tensor([[10, 20],
                          [30, 40]], dtype=torch.int32).numpy()
    mask = torch.ones((2, 2), dtype=torch.bool).numpy()
    source = torch.tensor([-1, -2, -3, -4], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 7: 1D complex64
    input = torch.tensor([1+2j, -3+0.5j, 0-1j], dtype=torch.complex64).numpy()
    mask = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    source = torch.tensor([2-2j, -1+1j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 8: 2D bool tensors
    input = torch.tensor([[True, False, True],
                          [False, False, True]], dtype=torch.bool).numpy()
    mask = torch.tensor([[False, True, False],
                         [True, False, True]], dtype=torch.bool).numpy()
    source = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 9: 3D float32 with NaN/Inf in source
    input = (torch.arange(12, dtype=torch.float32) - 5).view(3, 1, 4).numpy()
    mask_t = torch.zeros((3, 1, 4), dtype=torch.bool)
    mask_t[0, 0, 0] = True
    mask_t[0, 0, 3] = True
    mask_t[1, 0, 1] = True
    mask_t[2, 0, 0] = True
    mask_t[2, 0, 3] = True
    mask = mask_t.numpy()
    source = torch.tensor([float('nan'), -1.5, float('inf'), -float('inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 10: 3D int16
    input = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8]]], dtype=torch.int16).numpy()
    mask = torch.tensor([[[True, False],
                          [False, True]],
                         [[False, False],
                          [True, False]]], dtype=torch.bool).numpy()
    source = torch.tensor([-10, 20, -30], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 11: 1D float64, all True
    input = torch.tensor([0.5, -1.5, 2.5, -3.5, 4.5, -5.5, 6.5, -7.5], dtype=torch.float64).numpy()
    mask = torch.ones(8, dtype=torch.bool).numpy()
    source = torch.tensor([8.0, 7.0, 6.0, 5.0, -4.0, -3.0, -2.0, -1.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    # Input 12: 3D float32, partial True
    input = torch.tensor([[[1.0], [2.0], [3.0]],
                          [[-1.0], [-2.0], [-3.0]]], dtype=torch.float32).numpy()
    mask = torch.tensor([[[True], [False], [True]],
                         [[False], [True], [False]]], dtype=torch.bool).numpy()
    source = torch.tensor([9.9, -9.9, 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "mask": mask, "source": source}))

    return list_of_inputs

generated_inputs["torch.masked_scatter"] = masked_scatter_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.masked_scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.masked_scatter'.")


check_valid('torch.masked_scatter', generated_inputs['torch.masked_scatter'], lib="torch", suffix=0)
