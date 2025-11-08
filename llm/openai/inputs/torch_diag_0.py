
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def diag_inputs():
    list_of_inputs = []

    # 1) 1-D float32, k=0
    input_arr = torch.tensor([0.5, -1.2, 3.4], dtype=torch.float32).numpy()
    diagonal = np.int32(0)
    out = torch.zeros((3, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 2) 1-D float64, k=2
    input_arr = torch.tensor([1.0, -2.0, 3.0, 4.5, -5.5], dtype=torch.float64).numpy()
    diagonal = np.int64(2)
    out = torch.zeros((7, 7), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 3) 1-D int64, k=-1
    input_arr = torch.tensor([1, 0, -1, 8], dtype=torch.int64).numpy()
    diagonal = np.int8(-1)
    out = torch.zeros((5, 5), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 4) 1-D complex64, k=0
    input_arr = torch.tensor([1+2j, -3+0.5j], dtype=torch.complex64).numpy()
    diagonal = np.int16(0)
    out = torch.zeros((2, 2), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 5) 1-D bool, k=1
    input_arr = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    diagonal = np.int32(1)
    out = torch.zeros((4, 4), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 6) 2-D float32 square, k=0
    input_arr = torch.tensor([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]], dtype=torch.float32).numpy()
    diagonal = np.int64(0)
    out = torch.zeros((3,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 7) 2-D float64 rectangular, k=1
    input_arr = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                              [1.1, 1.2, 1.3, 1.4]], dtype=torch.float64).numpy()
    diagonal = np.int32(1)
    out = torch.zeros((2,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 8) 2-D int32, k=-2
    input_arr = torch.tensor([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [10, 11, 12],
                              [13, 14, 15]], dtype=torch.int32).numpy()
    diagonal = np.int8(-2)
    out = torch.zeros((3,), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 9) 2-D complex128 square, k=3
    input_arr = torch.tensor([[1+0j, 2+1j, 3+0j, 4+0j],
                              [5+0j, 6+0j, 7-1j, 8+0j],
                              [9+0j, 10+0j, 11+0j, 12+2j],
                              [13+0j, 14+0j, 15+0j, 16+0j]], dtype=torch.complex128).numpy()
    diagonal = np.int16(3)
    out = torch.zeros((1,), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 10) 2-D bool rectangular, k=-5 (empty result)
    input_arr = torch.tensor([[True, False],
                              [False, True],
                              [True, True]], dtype=torch.bool).numpy()
    diagonal = np.int32(-5)
    out = torch.zeros((0,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 11) 1-D empty float32, k=0
    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    diagonal = np.int64(0)
    out = torch.zeros((0, 0), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    # 12) 2-D float16 1x1, k=0
    input_arr = torch.tensor([[3.14]], dtype=torch.float16).numpy()
    diagonal = np.int32(0)
    out = torch.zeros((1,), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "diagonal": diagonal, "out": out}))

    return list_of_inputs

generated_inputs["torch.diag"] = diag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diag'.")


check_valid('torch.diag', generated_inputs['torch.diag'], lib="torch", suffix=0)
