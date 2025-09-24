
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def irfft_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = torch.randn(5, dtype=torch.complex64).numpy()
    n1 = 8
    dim1 = -1
    norm1 = "backward"
    out1 = np.empty(n1, dtype=np.float32)
    input_dict1 = {"input": input1, "n": n1, "dim": dim1, "norm": norm1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, different dim
    input2 = torch.randn(3, 4, dtype=torch.complex64).numpy()
    n2 = 6
    dim2 = 0
    norm2 = "forward"
    out2 = np.empty((3, n2), dtype=np.float32)
    input_dict2 = {"input": input2, "n": n2, "dim": dim2, "norm": norm2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: No specified n
    input3 = torch.randn(6, dtype=torch.complex64).numpy()
    n3 = None
    dim3 = -1
    norm3 = "ortho"
    out3 = np.empty(2 * (input3.shape[-1] - 1), dtype=np.float32)
    input_dict3 = {"input": input3, "n": n3, "dim": dim3, "norm": norm3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different dim value
    input4 = torch.randn(2, 5, dtype=torch.complex64).numpy()
    n4 = 7
    dim4 = 1
    norm4 = "backward"
    out4 = np.empty((2, n4), dtype=np.float32)
    input_dict4 = {"input": input4, "n": n4, "dim": dim4, "norm": norm4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor
    input5 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    n5 = 5
    dim5 = 2
    norm5 = "forward"
    out5 = np.empty((2, 3, n5), dtype=np.float32)
    input_dict5 = {"input": input5, "n": n5, "dim": dim5, "norm": norm5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: different norm
    input6 = torch.randn(7, dtype=torch.complex64).numpy()
    n6 = 10
    dim6 = -1
    norm6 = "ortho"
    out6 = np.empty(n6, dtype=np.float32)
    input_dict6 = {"input": input6, "n": n6, "dim": dim6, "norm": norm6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Large n
    input7 = torch.randn(4, dtype=torch.complex64).numpy()
    n7 = 20
    dim7 = -1
    norm7 = "backward"
    out7 = np.empty(n7, dtype=np.float32)
    input_dict7 = {"input": input7, "n": n7, "dim": dim7, "norm": norm7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 2D, dim = 1, small n
    input8 = torch.randn(4, 5, dtype=torch.complex64).numpy()
    n8 = 3
    dim8 = 1
    norm8 = "forward"
    out8 = np.empty((4, n8), dtype=np.float32)
    input_dict8 = {"input": input8, "n": n8, "dim": dim8, "norm": norm8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Different dimension
    input9 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    n9 = 6
    dim9 = 0
    norm9 = "ortho"
    out9 = np.empty((n9, 3, 4), dtype=np.float32)
    input_dict9 = {"input": input9, "n": n9, "dim": dim9, "norm": norm9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Single element input tensor
    input10 = torch.randn(1, dtype=torch.complex64).numpy()
    n10 = 2
    dim10 = -1
    norm10 = "backward"
    out10 = np.empty(n10, dtype=np.float32)
    input_dict10 = {"input": input10, "n": n10, "dim": dim10, "norm": norm10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.irfft"] = irfft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.irfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.irfft'.")

check_valid('torch.fft.irfft', generated_inputs['torch.fft.irfft'], lib="torch", suffix=0)
