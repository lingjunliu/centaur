
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def logsumexp_inputs():
    def expected_shape(shape, dims, keepdim):
        nd = len(shape)
        norm = [(d + nd) if d < 0 else d for d in dims]
        red = set(norm)
        if keepdim:
            return tuple(1 if i in red else shape[i] for i in range(nd))
        else:
            return tuple(shape[i] for i in range(nd) if i not in red)

    list_of_inputs = []

    # Input 1
    inp = torch.arange(-2.0, 3.0, dtype=torch.float32).numpy()
    dims = [0]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 2
    inp = torch.tensor([[1.0, -2.0, 0.5], [3.2, -0.1, -4.4]], dtype=torch.float64).numpy()
    dims = [1]
    keep = True
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 3
    inp = torch.randn(4, 5, 6, dtype=torch.float32).numpy()
    dims = [1, 2]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 4
    inp = torch.randn(2, 3, 4, 5, dtype=torch.float16).numpy()
    dims = [-1]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 5
    t = torch.randn(6, 7, dtype=torch.float32).t()
    inp = t.numpy()
    dims = [0]
    keep = True
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 6
    inp = torch.tensor([[float("nan"), 1e20], [float("-inf"), 3.0]], dtype=torch.float32).numpy()
    dims = [0, 1]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 7
    inp = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    dims = [-2, -5]
    keep = True
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 8
    inp = torch.empty(0, 3, dtype=torch.float64).numpy()
    dims = [1]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 9
    inp = torch.randn(3, 4, 5, dtype=torch.float32).numpy()
    dims = [2, 0]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 10
    inp = (torch.randn(3, 2, 4, 5, dtype=torch.float32) * 1000).numpy()
    dims = [1, 3]
    keep = True
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 11
    inp = torch.ones(1, 1, 1, dtype=torch.float16).numpy()
    dims = [0, 1, 2]
    keep = True
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 12
    inp = torch.randn(2, 0, 5, dtype=torch.float32).numpy()
    dims = [1]
    keep = True
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    # Input 13
    inp = torch.linspace(-5.0, 5.0, steps=24, dtype=torch.float64).reshape(2, 3, 4).numpy()
    dims = [-3, -1]
    keep = False
    out = torch.empty(expected_shape(inp.shape, dims, keep), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dims, "keepdim": keep, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.logsumexp_2"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.logsumexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.logsumexp_2'.")


check_valid('torch.special.logsumexp', generated_inputs['torch.special.logsumexp_2'], lib="torch", suffix=2)
