
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def exp2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input = torch.tensor([-3.5, -1.0, 0.0, 1.0, 3.25], dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64
    input = torch.tensor([[0.0, 1.0, -2.0],
                          [3.5, -4.5, 10.0]], dtype=torch.float64).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 0D scalar float32
    input = torch.tensor(1.5, dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 3D float32
    input = torch.tensor([[[ -2.0, -1.0, 0.0, 1.0 ],
                           [  2.0,  3.0, 4.0, -3.0 ]],
                          [[  0.5, -0.5, 6.0, -6.0 ],
                           [  8.0, -8.0, 0.25, -0.25 ]]], dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 4D float64 with extreme values
    base = torch.tensor([-1000.0, -100.0, 0.0, 10.0, 100.0, 1000.0], dtype=torch.float64)
    input = base.view(1, 2, 1, 3).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: empty array shape (0, 5) float32
    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: float16 1D
    input = torch.tensor([-5.0, -1.25, 0.0, 1.25, 5.0, 10.0], dtype=torch.float16).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: complex64 1D
    input = torch.tensor([1.0+1.0j, -2.0+0.5j, 0.0-3.0j], dtype=torch.complex64).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: complex128 2D
    input = torch.tensor([[0.0+0.0j, 2.0-1.0j],
                          [-3.0+4.0j, 1.5-2.5j]], dtype=torch.complex128).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: non-contiguous slice float32
    base = torch.arange(0, 12, dtype=torch.float32).view(3, 4).numpy()
    input = base[:, ::2]
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: 5D float32
    input = torch.tensor(0.1, dtype=torch.float32).expand(1, 2, 1, 3, 2).contiguous().numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: special values nan/inf float32
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0, 1.0], dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.exp2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.exp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.exp2'.")


check_valid('torch.special.exp2', generated_inputs['torch.special.exp2'], lib="torch", suffix=0)
