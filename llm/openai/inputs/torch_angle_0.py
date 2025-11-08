
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def angle_inputs():
    list_of_inputs = []

    # 1: 1D complex64 array with mixed quadrants
    inp = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j, 1 + 0j], dtype=torch.complex64).numpy()
    out = np.empty(inp.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 2: 1D real float32 with negatives, zeros, and NaN
    inp = torch.tensor([-0.0, 0.0, -1.0, 2.5, float('nan')], dtype=torch.float32).numpy()
    out = np.empty(inp.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 3: 2D complex128 matrix with various cases
    inp = torch.tensor([[1 + 2j, -3 + 0j], [0 - 4j, -5 - 6j]], dtype=torch.complex128).numpy()
    out = np.empty(inp.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 4: 2D real float64 matrix
    inp = torch.tensor([[1.0, -2.0, 0.0], [3.5, -4.5, 6.0]], dtype=torch.float64).numpy()
    out = np.empty(inp.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 5: 0D scalar complex64
    inp = torch.tensor(1 - 1j, dtype=torch.complex64).numpy()
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 6: 3D real float32 tensor
    inp = torch.tensor([[[-1.0, 0.0, 1.0]], [[2.0, -3.0, 4.0]]], dtype=torch.float32).numpy()
    out = np.empty(inp.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 7: 3D complex64 tensor
    inp = torch.tensor(
        [[[1 + 0j, 0 + 1j], [-1 + 0j, 0 - 1j]],
         [[1 + 1j, -1 - 1j], [2 - 2j, -2 + 2j]]],
        dtype=torch.complex64
    ).numpy()
    out = np.empty(inp.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 8: 1D real float64 with large/small magnitudes
    inp = torch.tensor([-1e20, 1e-20, -1e-30, 0.0, 3.14], dtype=torch.float64).numpy()
    out = np.empty(inp.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 9: 1D complex128 with NaN and Inf values
    inp = torch.tensor([complex(float('nan'), 1.0), complex(float('inf'), -float('inf')), complex(-float('inf'), 0.0)], dtype=torch.complex128).numpy()
    out = np.empty(inp.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 10: 2D real float16
    inp = torch.tensor([[0.0, -2.0], [3.0, -4.0]], dtype=torch.float16).numpy()
    out = np.empty(inp.shape, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 11: 1D complex64 points on the unit circle
    angles = [0.0, np.pi / 2, np.pi, -np.pi / 2, np.pi / 4, -3 * np.pi / 4]
    complex_vals = [np.cos(a) + 1j * np.sin(a) for a in angles]
    inp = torch.tensor(complex_vals, dtype=torch.complex64).numpy()
    out = np.empty(inp.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    # 12: 4D real float32 tensor
    inp = torch.tensor([[[[0.0, -1.0], [2.0, -3.0]]],
                        [[[4.0, -5.0], [6.0, -7.0]]]], dtype=torch.float32).numpy()
    out = np.empty(inp.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    return list_of_inputs

generated_inputs["torch.angle"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.angle'.")


check_valid('torch.angle', generated_inputs['torch.angle'], lib="torch", suffix=0)
