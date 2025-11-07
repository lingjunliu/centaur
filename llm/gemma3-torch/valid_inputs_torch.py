generated_inputs = {}
import torch, copy
import numpy as np

def addcdiv_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    tensor1_1 = torch.randn(2, 3).numpy()
    tensor2_1 = torch.randn(2, 3).numpy()
    value1 = 0.5
    out1 = torch.empty(2, 3).numpy()

    input_dict1 = {
        "input": input1,
        "tensor1": tensor1_1,
        "tensor2": tensor2_1,
        "value": value1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1_2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2_2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    value2 = 2.0
    out2 = torch.empty(3).numpy()

    input_dict2 = {
        "input": input2,
        "tensor1": tensor1_2,
        "tensor2": tensor2_2,
        "value": value2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 2).numpy()
    tensor1_3 = torch.randn(3, 2).numpy()
    tensor2_3 = torch.randn(3, 2).numpy()
    value3 = -0.1
    out3 = torch.empty(3, 2).numpy()

    input_dict3 = {
        "input": input3,
        "tensor1": tensor1_3,
        "tensor2": tensor2_3,
        "value": value3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.ones((1, 4)).numpy()
    tensor1_4 = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).numpy()
    tensor2_4 = torch.tensor([[2.0, 2.0, 2.0, 2.0]]).numpy()
    value4 = 1.0
    out4 = torch.empty((1, 4)).numpy()

    input_dict4 = {
        "input": input4,
        "tensor1": tensor1_4,
        "tensor2": tensor2_4,
        "value": value4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.addcdiv"] = addcdiv_inputs()

import torch, copy
import numpy as np

def addmv_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    vec1 = np.array([7.0, 8.0], dtype=np.float32)
    beta1 = 1.0
    alpha1 = 1.0
    out1 = np.array([], dtype=np.float32)
    
    input_dict1 = {
        "input": input1,
        "mat": mat1,
        "vec": vec1,
        "beta": beta1,
        "alpha": alpha1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    mat2 = np.array([[0.5, -0.5], [1.0, 0.0], [-0.5, 0.5]], dtype=np.float64)
    vec2 = np.array([2.0, -1.0], dtype=np.float64)
    beta2 = 0.5
    alpha2 = 2.0
    out2 = np.array([], dtype=np.float64)

    input_dict2 = {
        "input": input2,
        "mat": mat2,
        "vec": vec2,
        "beta": beta2,
        "alpha": alpha2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 1.0], dtype=np.float32)
    mat3 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    vec3 = np.array([0.5, 0.5], dtype=np.float32)
    beta3 = 1.0
    alpha3 = 1.0
    out3 = np.array([], dtype=np.float32)

    input_dict3 = {
        "input": input3,
        "mat": mat3,
        "vec": vec3,
        "beta": beta3,
        "alpha": alpha3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat4 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    vec4 = np.array([0.5, 1.0], dtype=np.float32)
    beta4 = 1.0
    alpha4 = 1.0
    out4 = np.array([], dtype=np.float32)

    input_dict4 = {
        "input": input4,
        "mat": mat4,
        "vec": vec4,
        "beta": beta4,
        "alpha": alpha4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0], dtype=np.float32)
    mat5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vec5 = np.array([0.5, 1.0], dtype=np.float32)
    beta5 = 1.0
    alpha5 = 1.0
    out5 = np.array([], dtype=np.float32)

    input_dict5 = {
        "input": input5,
        "mat": mat5,
        "vec": vec5,
        "beta": beta5,
        "alpha": alpha5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.addmv"] = addmv_inputs()

import torch, copy
import numpy as np

def amin_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dim1 = 0
    keepdim1 = True
    out1 = np.empty((1, 3), dtype=np.float64)

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float64)
    dim2 = 1
    keepdim2 = False
    out2 = np.empty((2,), dtype=np.float64)

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(3, 4, 5).astype(np.float64)
    dim3 = (0, 1)
    keepdim3 = True
    out3 = np.empty((1, 1, 5), dtype=np.float64)

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.5, 0.2], [0.8, 0.1]], dtype=np.float64)
    dim4 = None
    keepdim4 = False
    out4 = np.empty(() , dtype=np.float64)

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([10], dtype=np.float64)
    dim5 = 0
    keepdim5 = True
    out5 = np.empty((1,), dtype=np.float64)

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    dim6 = 0
    keepdim6 = False
    out6 = np.empty((2,), dtype=np.float64)

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "keepdim": keepdim6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    dim7 = (1, 3)
    keepdim7 = False
    out7 = np.empty((2, 2), dtype=np.float64)

    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "keepdim": keepdim7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs["torch.amin"] = amin_inputs()

import torch, copy
import numpy as np

def any_inputs():
    list_of_inputs = []

    input1 = np.array([True, False, True], dtype=bool)
    dim1 = None
    keepdim1 = False
    out1 = np.array([], dtype=bool)

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.any_2"] = any_inputs()

import torch, copy

def are_deterministic_algorithms_enabled_inputs():
    list_of_inputs = []

    input1 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input1}))

    input2 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input2}))

    input3 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input3}))

    input4 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input4}))

    input5 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input5}))
    
    input6 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input6}))
    
    input7 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input7}))

    input8 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input8}))

    input9 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input9}))

    input10 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input10}))
    
    return list_of_inputs

generated_inputs["torch.are_deterministic_algorithms_enabled"] = are_deterministic_algorithms_enabled_inputs()

import torch, copy
import numpy as np

def asin_inputs():
    list_of_inputs = []
    
    input1 = np.array([-1.0, 0.0, 1.0])
    out1 = np.empty_like(input1)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-0.5, 0.5, -1.0])
    out2 = np.empty_like(input2)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([0.0, 0.0, 0.0])
    out3 = np.empty_like(input3)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1.0, 0.0, 1.0, -0.7, 0.7])
    out4 = np.empty_like(input4)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    out5 = np.empty_like(input5)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[-1.0, 0.0], [1.0, -1.0]])
    out6 = np.empty_like(input6)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[[0.0, 1.0], [-1.0, 0.0]], [[0.0, -1.0], [1.0, 0.0]]])
    out7 = np.empty_like(input7)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([-1.0, 1.0, 0.0, -0.5, 0.5], dtype=np.float64)
    out8 = np.empty_like(input8, dtype=np.float64)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    out9 = np.empty_like(input9, dtype=np.float32)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.2, -0.8, 0.5, -0.1])
    out10 = np.empty_like(input10)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.asin"] = asin_inputs()

import torch, copy
import numpy as np

def bincount_inputs():
    list_of_inputs = []
    
    input1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    weights1 = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    minlength1 = 0
    
    input_dict1 = {
        "input": input1,
        "weights": weights1,
        "minlength": minlength1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    weights2 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    minlength2 = 3
    
    input_dict2 = {
        "input": input2,
        "weights": weights2,
        "minlength": minlength2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([], dtype=np.int64)
    weights3 = None
    minlength3 = 5
    
    input_dict3 = {
        "input": input3,
        "weights": weights3,
        "minlength": minlength3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    weights4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    minlength4 = 0
    
    input_dict4 = {
        "input": input4,
        "weights": weights4,
        "minlength": minlength4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([0, 1, 0, 2, 1, 0], dtype=np.int64)
    weights5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    minlength5 = 3
    
    input_dict5 = {
        "input": input5,
        "weights": weights5,
        "minlength": minlength5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([7, 7, 7, 7], dtype=np.int64)
    weights6 = np.array([1, 1, 1, 1], dtype=np.float32)
    minlength6 = 8
    
    input_dict6 = {
        "input": input6,
        "weights": weights6,
        "minlength": minlength6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    weights7 = np.array([1, 2, 1, 2, 1, 2], dtype=np.float32)
    minlength7 = 0

    input_dict7 = {
        "input": input7,
        "weights": weights7,
        "minlength": minlength7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([10], dtype=np.int64)
    weights8 = np.array([1.0], dtype=np.float32)
    minlength8 = 5
    
    input_dict8 = {
        "input": input8,
        "weights": weights8,
        "minlength": minlength8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    weights9 = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    minlength9 = 10
    
    input_dict9 = {
        "input": input9,
        "weights": weights9,
        "minlength": minlength9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    weights10 = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    minlength10 = 2

    input_dict10 = {
        "input": input10,
        "weights": weights10,
        "minlength": minlength10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.bincount"] = bincount_inputs()

import torch, copy
import numpy as np

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([1, 2, 3]).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([0, 1, 2]).numpy()
    other2 = torch.tensor([4, 5, 6]).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([-1, -2, -3]).numpy()
    other3 = torch.tensor([1, 2, 3]).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([10, 20, 30]).numpy()
    other4 = torch.tensor([0, 1, 2]).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other5 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other6 = torch.tensor([2, 2, 2, 2, 2]).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other7 = torch.tensor([1, 2, 3]).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([1, 2]).numpy()
    other8 = torch.tensor([5, 6]).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.tensor([0, 0, 0]).numpy()
    other9 = torch.tensor([1, 2, 3]).numpy()
    input_dict9 = {"input": input9, "other": other9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([1, 1, 1]).numpy()
    other10 = torch.tensor([0, 0, 0]).numpy()
    input_dict10 = {"input": input10, "other": other10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

import torch, copy
import numpy as np

def clip_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, -2.0, 3.0])
    min1 = np.array([0.0, -1.0, 2.0])
    max1 = np.array([2.0, 0.0, 4.0])
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "min": min1,
        "max": max1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3)
    min2 = np.array([-1.0])
    max2 = np.array([1.0])
    out2 = np.array([])
    
    input_dict2 = {
        "input": input2,
        "min": min2,
        "max": max2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.clip"] = clip_inputs()

import torch, copy
import numpy as np

def torch_complex_inputs():
    list_of_inputs = []

    real1 = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()
    imag1 = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()
    out1 = np.empty((2,), dtype=np.complex64)
    input_dict1 = {"real": real1, "imag": imag1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    real2 = torch.tensor([-1.0, 2.0], dtype=torch.float64).numpy()
    imag2 = torch.tensor([3.0, -4.0], dtype=torch.float64).numpy()
    out2 = np.empty((2,), dtype=np.complex128)
    input_dict2 = {"real": real2, "imag": imag2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    real3 = torch.randn(3, 2, dtype=torch.float32).numpy()
    imag3 = torch.randn(3, 2, dtype=torch.float32).numpy()
    out3 = np.empty((3, 2), dtype=np.complex64)
    input_dict3 = {"real": real3, "imag": imag3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    real4 = torch.tensor([0.0], dtype=torch.float32).numpy()
    imag4 = torch.tensor([0.0], dtype=torch.float32).numpy()
    out4 = np.empty((1,), dtype=np.complex64)
    input_dict4 = {"real": real4, "imag": imag4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    real5 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32).numpy()
    imag5 = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32).numpy()
    out5 = np.empty((3,), dtype=np.complex64)
    input_dict5 = {"real": real5, "imag": imag5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    real6 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    imag6 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64).numpy()
    out6 = np.empty((3,), dtype=np.complex128)
    input_dict6 = {"real": real6, "imag": imag6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    real7 = torch.zeros(2, 2, dtype=torch.float32).numpy()
    imag7 = torch.ones(2, 2, dtype=torch.float32).numpy()
    out7 = np.empty((2, 2), dtype=np.complex64)
    input_dict7 = {"real": real7, "imag": imag7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    real8 = torch.arange(1, 5, dtype=torch.float32).numpy()
    imag8 = torch.arange(5, 9, dtype=torch.float32).numpy()
    out8 = np.empty((4,), dtype=np.complex64)
    input_dict8 = {"real": real8, "imag": imag8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    real9 = torch.tensor([1e-6, 2e-6], dtype=torch.float32).numpy()
    imag9 = torch.tensor([3e-6, 4e-6], dtype=torch.float32).numpy()
    out9 = np.empty((2,), dtype=np.complex64)
    input_dict9 = {"real": real9, "imag": imag9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    real10 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    imag10 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64).numpy()
    out10 = np.empty((4,), dtype=np.complex128)
    input_dict10 = {"real": real10, "imag": imag10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.complex"] = torch_complex_inputs()

import torch, copy
import numpy as np

def conj_physical_inputs():
    list_of_inputs = []
    
    input1 = np.array([1+1j, 2+2j, 3+3j])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1-1j, 2+2j], [3-3j, 4+4j]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1.0+1.0j, 2.0-2.0j, 3.0+0.0j])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.conj_physical"] = conj_physical_inputs()

import torch, copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(3)
    list_of_inputs.append({"tensor": input1.numpy()})

    input2 = torch.randn(2, 4)
    list_of_inputs.append({"tensor": input2.numpy()})

    input3 = torch.randn(5, 5, 5)
    list_of_inputs.append({"tensor": input3.numpy()})

    input4 = torch.randn(2, 2)
    list_of_inputs.append({"tensor": input4.numpy()})

    input5 = torch.randn(4)
    list_of_inputs.append({"tensor": input5.numpy()})
    
    input6 = torch.randn(10, 10, 10, 10)
    list_of_inputs.append({"tensor": input6.numpy()})

    input7 = torch.randn(3, 3)
    list_of_inputs.append({"tensor": input7.numpy()})
    
    input8 = torch.randn(2, 3, 4)
    list_of_inputs.append({"tensor": input8.numpy()})

    input9 = torch.randn(6, 6)
    list_of_inputs.append({"tensor": input9.numpy()})

    input10 = torch.randn(7)
    list_of_inputs.append({"tensor": input10.numpy()})
    
    return list_of_inputs

generated_inputs["torch.dequantize_1"] = dequantize_inputs()

import torch, copy
import numpy as np

def diag_embed_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3]).astype(np.int32)
    offset1 = 0
    dim1_1 = 0
    dim2_1 = 1
    input_dict1 = {
        "input": input1,
        "offset": offset1,
        "dim1": dim1_1,
        "dim2": dim2_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3, 4]).astype(np.float64)
    offset2 = 1
    dim1_2 = 0
    dim2_2 = 1
    input_dict2 = {
        "input": input2,
        "offset": offset2,
        "dim1": dim1_2,
        "dim2": dim2_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 2], [3, 4]]).astype(np.int64)
    offset3 = -1
    dim1_3 = 0
    dim2_3 = 1
    input_dict3 = {
        "input": input3,
        "offset": offset3,
        "dim1": dim1_3,
        "dim2": dim2_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    offset4 = 2
    dim1_4 = 1
    dim2_4 = 2
    input_dict4 = {
        "input": input4,
        "offset": offset4,
        "dim1": dim1_4,
        "dim2": dim2_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3, 4, 5]).astype(np.int16)
    offset5 = 0
    dim1_5 = 0
    dim2_5 = 1
    input_dict5 = {
        "input": input5,
        "offset": offset5,
        "dim1": dim1_5,
        "dim2": dim2_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]]).astype(np.complex128)
    offset6 = -2
    dim1_6 = 0
    dim2_6 = 1
    input_dict6 = {
        "input": input6,
        "offset": offset6,
        "dim1": dim1_6,
        "dim2": dim2_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.diag_embed"] = diag_embed_inputs()

import torch, copy
import numpy as np

def empty_like_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    dtype1 = torch.float32
    layout1 = torch.strided
    device1 = 'cpu'
    requires_grad1 = False
    
    input_dict1 = {
        "input": input1,
        "dtype": dtype1,
        "layout": layout1,
        "device": device1,
        "requires_grad": requires_grad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.empty_like"] = empty_like_inputs()

import torch, copy
import numpy as np

def equal_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([1, 2, 3]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([-1, -2, -3]).numpy()
    input2 = torch.tensor([-1, -2, -3]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1, 2]).numpy()
    input2 = torch.tensor([1, 2]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    input2 = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1, 2, np.nan]).numpy()
    input2 = torch.tensor([1, 2, np.nan]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([1, 2, 4]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1]).numpy()
    input2 = torch.tensor([1]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    input1 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input2 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    list_of_inputs.append({"input": input1, "other": input2})

    return list_of_inputs

generated_inputs["torch.equal"] = equal_inputs()

import torch, copy
import numpy as np

def fix_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1.5, 2.7, 3.1]).numpy()
    out1 = torch.tensor([]) # Corrected: Initialize tensor with no data
    input_dict1 = {"input": input1, "out": out1.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.5, -2.7, -3.1]).numpy()
    out2 = torch.tensor([])
    input_dict2 = {"input": input2, "out": out2.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    out3 = torch.tensor([])
    input_dict3 = {"input": input3, "out": out3.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3).numpy()
    out4 = torch.tensor([])
    input_dict4 = {"input": input4, "out": out4.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 2, 4).numpy()
    out5 = torch.tensor([])
    input_dict5 = {"input": input5, "out": out5.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

import torch, copy
import numpy as np

def flipud_inputs():
    list_of_inputs = []
    
    input1 = torch.arange(4).view(2, 2).numpy()
    input_dict = {"input": input1}
    list_of_inputs.append(input_dict)

    input2 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    input_dict = {"input": input2}
    list_of_inputs.append(input_dict)
    
    input3 = torch.randn(3, 4).numpy()
    input_dict = {"input": input3}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["torch.flipud"] = flipud_inputs()

import torch, copy
import numpy as np

def float_power_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    exponent1 = 2.0
    input_dict1 = {"input": input1, "exponent": exponent1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    exponent2 = 0.5
    input_dict2 = {"input": input2, "exponent": exponent2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([-1.0, -2.0, -3.0])
    exponent3 = 3.0
    input_dict3 = {"input": input3, "exponent": exponent3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0, 1.0, 2.0])
    exponent4 = -1.0
    input_dict4 = {"input": input4, "exponent": exponent4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.5, 2.5, 3.5])
    exponent5 = 1.5
    input_dict5 = {"input": input5, "exponent": exponent5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    exponent6 = 2.0
    input_dict6 = {"input": input6, "exponent": exponent6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0])
    exponent7 = 0.0
    input_dict7 = {"input": input7, "exponent": exponent7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([2.0, 4.0, 8.0])
    exponent8 = -0.5
    input_dict8 = {"input": input8, "exponent": exponent8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.1, 2.2, 3.3])
    exponent9 = 1.0
    input_dict9 = {"input": input9, "exponent": exponent9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.5, 1.5, 2.5])
    exponent10 = 2.5
    input_dict10 = {"input": input10, "exponent": exponent10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.float_power"] = float_power_inputs()

import torch, copy
import numpy as np

def floor_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.2, -2.5, 3.8, -0.1]).numpy()
    out1 = torch.empty(0).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.7, 0.0, 2.9, -3.3]).numpy()
    out2 = torch.empty(0).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 2).numpy()
    out3 = torch.empty(0).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[1.5, -2.5], [3.1, -0.9]]).numpy()
    out4 = torch.empty(0).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([[-1.2, 0.5, 2.7], [3.8, -0.1, -2.3]]).numpy()
    out5 = torch.empty(0).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([5.0, 2.0, 1.0]).numpy()
    out6 = torch.empty(0).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([-5.0, -2.0, -1.0]).numpy()
    out7 = torch.empty(0).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.floor"] = floor_inputs()

import torch, copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([10, 20, 30]).numpy()
    other1 = torch.tensor([2, 4, 6]).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-10, -20, -30]).numpy()
    other2 = torch.tensor([2, 4, 6]).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([10.5, 20.5, 30.5]).numpy()
    other3 = torch.tensor([2.0, 4.0, 6.0]).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[10, 20], [30, 40]]).numpy()
    other4 = torch.tensor([[2, 4], [6, 8]]).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other5 = torch.tensor([1, 1, 1, 1, 1]).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([10, 20, 30]).numpy()
    other6 = torch.tensor([3, 3, 3]).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other7 = torch.tensor([[0.5, 1.0], [1.5, 2.0]]).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    other8 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.tensor([10, 5, 2]).numpy()
    other9 = torch.tensor([3, 1, 1]).numpy()
    input_dict9 = {"input": input9, "other": other9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.floor_divide"] = floor_divide_inputs()

import torch, copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([10, 20, 30]).numpy()
    other1 = np.int32(2)
    
    input_dict1 = {
        "input": input1,
        "other": other1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-10, -20, 30]).numpy()
    other2 = np.int32(3)
    
    input_dict2 = {
        "input": input2,
        "other": other2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([10.5, 20.5, 30.5]).numpy()
    other3 = np.int32(2)
    
    input_dict3 = {
        "input": input3,
        "other": other3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[10, 20], [30, 40]]).numpy()
    other4 = np.int32(5)
    
    input_dict4 = {
        "input": input4,
        "other": other4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    other5 = np.int32(1)
    
    input_dict5 = {
        "input": input5,
        "other": other5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([10, 20, 30]).numpy()
    other6 = np.int64(4)

    input_dict6 = {
        "input": input6,
        "other": other6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other7 = np.int32(2)

    input_dict7 = {
        "input": input7,
        "other": other7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([100, 200, 300]).numpy()
    other8 = np.int32(10)
    
    input_dict8 = {
        "input": input8,
        "other": other8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

import torch, copy
import numpy as np

def torch_ge_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([1, 2, 3]).numpy()
    out1 = torch.tensor([False, False, False]).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other2 = torch.tensor([[1, 1], [4, 4]]).numpy()
    out2 = torch.tensor([[True, True], [False, True]]).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[-1, -2], [3, 4]]).numpy()
    other3 = torch.tensor([0, 0]).numpy()
    out3 = torch.tensor([False, False, True, True]).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out4 = torch.tensor([True, True, True]).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2).numpy()
    other5 = torch.randn(2, 2).numpy()
    out5 = torch.zeros((2, 2)).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([5]).numpy()
    other6 = torch.tensor([2]).numpy()
    out6 = torch.tensor([True]).numpy()
    input_dict6 = {"input": input6, "other": other6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 2, 3, 4]).numpy()
    other7 = torch.tensor([4, 3, 2, 1]).numpy()
    out7 = torch.tensor([False, False, True, True]).numpy()
    input_dict7 = {"input": input7, "other": other7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([0, 0, 0]).numpy()
    other8 = torch.tensor([0, 0, 0]).numpy()
    out8 = torch.tensor([True, True, True]).numpy()
    input_dict8 = {"input": input8, "other": other8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([[-1, -2, -3]]).numpy()
    other9 = torch.tensor([[-2, -1, 0]]).numpy()
    out9 = torch.tensor([[True, True, False]]).numpy()
    input_dict9 = {"input": input9, "other": other9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([1, 2, 3]).numpy()
    other10 = torch.tensor([1, 2, 3]).numpy()
    out10 = torch.tensor([True, True, True]).numpy()
    input_dict10 = {"input": input10, "other": other10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.ge"] = torch_ge_inputs()

import torch, copy
import numpy as np

def gradient_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0, 4.0])
    spacing1 = 1.0
    dim1 = 0
    edge_order1 = 1
    input_dict1 = {
        "input": input1,
        "spacing": spacing1,
        "dim": dim1,
        "edge_order": edge_order1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.gradient"] = gradient_inputs()

import torch, copy
import numpy as np

def imag_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(4, dtype=torch.cfloat).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 2, dtype=torch.cfloat).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(3, dtype=torch.cfloat).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(5, dtype=torch.cfloat).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 1, 1, dtype=torch.cfloat).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(2, 3, dtype=torch.cfloat).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(10, dtype=torch.cfloat).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(4, 4, dtype=torch.cfloat).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(2, 2, 2, dtype=torch.cfloat).numpy()
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(6, dtype=torch.cfloat).numpy()
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.imag"] = imag_inputs()

import torch, copy
import numpy as np

def is_floating_point_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.is_floating_point"] = is_floating_point_inputs()

import torch, copy
import numpy as np

def is_grad_enabled_inputs():
    list_of_inputs = []
    
    input1 = True
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input1}))
    
    input2 = False
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input2}))
    
    input3 = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input3}))

    input4 = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input4}))

    input5 = 1
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input5}))

    input6 = 0
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input6}))

    input7 = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input7}))

    input8 = np.int8(0)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input8}))

    input9 = np.uint8(1)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input9}))

    input10 = np.uint8(0)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input10}))
    
    input11 = np.int16(1)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input11}))

    input12 = np.int16(0)
    list_of_inputs.append(copy.deepcopy({"torch.is_grad_enabled": input12}))

    return list_of_inputs

generated_inputs["torch.is_grad_enabled"] = is_grad_enabled_inputs()

import torch, copy
import numpy as np

def lcm_inputs():
    list_of_inputs = []
    
    input1 = np.array([5, 10, 15], dtype=np.int64)
    other1 = np.array([3, 4, 5], dtype=np.int64)
    out1 = np.array([], dtype=np.int64)
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([2, 4, 6], dtype=np.int32)
    other2 = np.array([1, 3, 5], dtype=np.int32)
    out2 = np.array([], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-5, 10, -15], dtype=np.int64)
    other3 = np.array([3, -4, 5], dtype=np.int64)
    out3 = np.array([], dtype=np.int64)
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other4 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    out4 = np.array([], dtype=np.int64)
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0, 5, 10], dtype=np.int64)
    other5 = np.array([3, 0, 6], dtype=np.int64)
    out5 = np.array([], dtype=np.int64)
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([12, 18, 24], dtype=np.int32)
    other6 = np.array([8, 12, 16], dtype=np.int32)
    out6 = np.array([], dtype=np.int32)
    input_dict6 = {"input": input6, "other": other6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([7], dtype=np.int64)
    other7 = np.array([13], dtype=np.int64)
    out7 = np.array([], dtype=np.int64)
    input_dict7 = {"input": input7, "other": other7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1, 1, 1], dtype=np.int16)
    other8 = np.array([2, 3, 5], dtype=np.int16)
    out8 = np.array([], dtype=np.int16)
    input_dict8 = {"input": input8, "other": other8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([2**31 - 1, 2**31 - 1], dtype=np.int32)
    other9 = np.array([3, 5], dtype=np.int32)
    out9 = np.array([], dtype=np.int32)
    input_dict9 = {"input": input9, "other": other9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100, 200, 300], dtype=np.int64)
    other10 = np.array([50, 100, 150], dtype=np.int64)
    out10 = np.array([], dtype=np.int64)
    input_dict10 = {"input": input10, "other": other10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.lcm"] = lcm_inputs()

import torch, copy
import numpy as np

def svdvals_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(5, 3).numpy()
    driver1 = None
    out1 = np.zeros((3,), dtype=np.float32)
    
    input_dict1 = {
        "A": input1,
        "driver": driver1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 2).numpy()
    driver2 = None
    out2 = np.zeros((2,), dtype=np.float32)
    
    input_dict2 = {
        "A": input2,
        "driver": driver2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 5).numpy()
    driver3 = None
    out3 = np.zeros((5,), dtype=np.float32)
    
    input_dict3 = {
        "A": input3,
        "driver": driver3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(4, 4).numpy()
    driver4 = None
    out4 = np.zeros((4,), dtype=np.float32)
    
    input_dict4 = {
        "A": input4,
        "driver": driver4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 1).numpy()
    driver5 = None
    out5 = np.zeros((1,), dtype=np.float32)
    
    input_dict5 = {
        "A": input5,
        "driver": driver5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.linalg.svdvals"] = svdvals_inputs()

import torch, copy
import numpy as np

def log1p_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    out1 = np.array([])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-0.5, 0.0, 0.5])
    out2 = np.array([0.0])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    out3 = np.array([[0.0, 0.0], [0.0, 0.0]])
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1.0, -2.0, -3.0])
    out4 = np.array([])
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.001, 0.002, 0.003])
    out5 = np.array([])
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([10.0, 20.0, 30.0])
    out6 = np.array([])
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.0], [0.0], [1.0]])
    out7 = np.array([])
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([np.nan, 1.0, 2.0])
    out8 = np.array([])
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.0e-10, 1.0e-9, 1.0e-8])
    out9 = np.array([])
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([])
    out10 = np.array([])
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.log1p"] = log1p_inputs()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    keepdim1 = False
    out1 = np.array([], dtype=np.float32)
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dim2 = 1
    keepdim2 = True
    out2 = np.array([], dtype=np.float32)
    
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    dim3 = 0
    keepdim3 = False
    out3 = np.array([], dtype=np.float32)
    
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    dim4 = 2
    keepdim4 = True
    out4 = np.array([], dtype=np.float32)
    
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    dim5 = 1
    keepdim5 = False
    out5 = np.array([], dtype=np.float64)
    
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0], dtype=np.float16)
    dim6 = 0
    keepdim6 = True
    out6 = np.array([], dtype=np.float16)

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "keepdim": keepdim6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    dim7 = (0, 1)
    keepdim7 = False
    out7 = np.array([], dtype=np.float32)

    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "keepdim": keepdim7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    dim8 = 0
    keepdim8 = True
    out8 = np.array([], dtype=np.float32)

    input_dict8 = {
        "input": input8,
        "dim": dim8,
        "keepdim": keepdim8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dim9 = 0
    keepdim9 = False
    out9 = np.array([], dtype=np.float32)

    input_dict9 = {
        "input": input9,
        "dim": dim9,
        "keepdim": keepdim9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dim10 = 1
    keepdim10 = True
    out10 = np.array([], dtype=np.float32)

    input_dict10 = {
        "input": input10,
        "dim": dim10,
        "keepdim": keepdim10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.logsumexp"] = logsumexp_inputs()

import torch, copy
import numpy as np

def matrix_power_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n1 = 2
    input_dict1 = {"input": input1, "n": n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    n2 = 5
    input_dict2 = {"input": input2, "n": n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    n3 = 3
    input_dict3 = {"input": input3, "n": n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n4 = 0
    input_dict4 = {"input": input4, "n": n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[0.0, 1.0], [1.0, 0.0]])
    n5 = 4
    input_dict5 = {"input": input5, "n": n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[2.0, 1.0], [1.0, 2.0]])
    n6 = 2
    input_dict6 = {"input": input6, "n": n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[0.5, 0.2], [0.1, 0.8]])
    n7 = 3
    input_dict7 = {"input": input7, "n": n7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1.1, 2.2], [3.3, 4.4]])
    n8 = 2
    input_dict8 = {"input": input8, "n": n8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n9 = -1
    input_dict9 = {"input": input9, "n": n9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.matrix_power"] = matrix_power_inputs()

import torch, copy
import numpy as np

def median_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, 0.0, 1.0])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 1.0, 1.0])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.5, 2.5, 3.5, 4.5])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.median_1"] = median_inputs()

import torch, copy
import numpy as np

def minimum_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([4, 5, 6]).numpy()
    out1 = torch.tensor([0, 0, 0], dtype=torch.long).numpy()
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([-1, -2, -3]).numpy()
    other2 = torch.tensor([0, 1, 2]).numpy()
    out2 = torch.tensor([0, 0, 0], dtype=torch.long).numpy()
    
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs["torch.minimum"] = minimum_inputs()

import torch, copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    dtype1 = torch.float64
    input_dict1 = {"input": input1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, np.nan, 3.0], [4.0, -np.inf, 6.0]], dtype=np.float32)
    dtype2 = torch.float16
    input_dict2 = {"input": input2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    dtype3 = torch.float32
    input_dict3 = {"input": input3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=np.float16)
    dtype4 = torch.float32
    input_dict4 = {"input": input4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    dtype5 = None
    input_dict5 = {"input": input5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]], dtype=np.float32)
    dtype6 = torch.float64
    input_dict6 = {"input": input6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([-1.0, -2.0, np.nan, -4.0], dtype=np.float32)
    dtype7 = torch.float16
    input_dict7 = {"input": input7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dtype8 = torch.float32
    input_dict8 = {"input": input8, "dtype": dtype8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    dtype9 = torch.float64
    input_dict9 = {"input": input9, "dtype": dtype9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    dtype10 = torch.float64
    input_dict10 = {"input": input10, "dtype": dtype10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

import torch, copy
import numpy as np

def avgpool1d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(1, 3, 10).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(1, 5, 15).numpy()
    input_dict2 = {
        "kernel_size": 5,
        "stride": 3,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(2, 4, 8).numpy()
    input_dict3 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(1, 2, 20).numpy()
    input_dict4 = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(3, 1, 12).numpy()
    input_dict5 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 4, 7).numpy()
    input_dict6 = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 3, 11).numpy()
    input_dict7 = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(1, 6, 18).numpy()
    input_dict8 = {
        "kernel_size": 5,
        "stride": 5,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(4, 2, 9).numpy()
    input_dict9 = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d"] = avgpool1d_inputs()

import torch, copy
import numpy as np

def avgpool2d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d"] = avgpool2d_inputs()

import torch, copy
import numpy as np

def bcewithlogitsloss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(10, 64).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(10, 64)).astype(np.float32)
    pos_weight1 = np.ones(64).astype(np.float32)
    
    input_dict1 = {
        "weight": np.array([1.0]).astype(np.float32),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "pos_weight": pos_weight1,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.nn.BCEWithLogitsLoss"] = bcewithlogitsloss_inputs()

import torch, copy
import numpy as np

def feature_alpha_dropout_inputs():
    list_of_inputs = []

    input1 = np.random.rand(20, 16, 4, 32, 32).astype(np.float32)
    input_dict1 = {"p": 0.2, "inplace": False, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(10, 8, 2, 16, 16).astype(np.float32)
    input_dict2 = {"p": 0.5, "inplace": True, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(5, 32, 1, 8, 8).astype(np.float32)
    input_dict3 = {"p": 0.1, "inplace": False, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 64, 8, 4, 4).astype(np.float32)
    input_dict4 = {"p": 0.9, "inplace": True, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(32, 16, 2, 4, 4).astype(np.float32)
    input_dict5 = {"p": 0.0, "inplace": False, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(16, 3, 1, 1).astype(np.float32)
    input_dict6 = {"p": 0.7, "inplace": True, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(8, 64, 4, 4).astype(np.float32)
    input_dict7 = {"p": 0.3, "inplace": False, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(4, 128, 2, 2).astype(np.float32)
    input_dict8 = {"p": 1.0, "inplace": True, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(64, 3, 16, 16).astype(np.float32)
    input_dict9 = {"p": 0.6, "inplace": False, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 1, 32, 32).astype(np.float32)
    input_dict10 = {"p": 0.8, "inplace": True, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.FeatureAlphaDropout"] = feature_alpha_dropout_inputs()

import torch, copy
import numpy as np

def huber_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.1, 1.9, 3.2], dtype=np.float32)
    input_dict1 = {"reduction": "mean", "delta": 1.0, "input": input1, "target": target1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.2, 1.8], [2.9, 4.1]], dtype=np.float32)
    input_dict2 = {"reduction": "sum", "delta": 0.5, "input": input2, "target": target2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    target3 = np.array([-1.1, -2.1, -2.9], dtype=np.float32)
    input_dict3 = {"reduction": "none", "delta": 2.0, "input": input3, "target": target3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    target4 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict4 = {"reduction": "mean", "delta": 0.1, "input": input4, "target": target4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    target5 = np.array([5.0, 15.0, 25.0], dtype=np.float32)
    input_dict5 = {"reduction": "sum", "delta": 5.0, "input": input5, "target": target5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0], dtype=np.float32)
    target6 = np.array([2.0], dtype=np.float32)
    input_dict6 = {"reduction": "mean", "delta": 1.0, "input": input6, "target": target6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4, 4).astype(np.float32)
    target7 = np.random.rand(4, 4, 4).astype(np.float32)
    input_dict7 = {"reduction": "none", "delta": 1.5, "input": input7, "target": target7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    target8 = np.array([0.0, -1.0, 4.0], dtype=np.float32)
    input_dict8 = {"reduction": "mean", "delta": 0.2, "input": input8, "target": target8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict9 = {"reduction": "sum", "delta": 10.0, "input": input9, "target": target9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([0.5, -0.5, 1.5], dtype=np.float32)
    target10 = np.array([0.6, -0.4, 1.4], dtype=np.float32)
    input_dict10 = {"reduction": "mean", "delta": 0.3, "input": input10, "target": target10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.HuberLoss"] = huber_loss_inputs()

import torch, copy
import numpy as np

def logsoftmax_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    input_dict1 = {"dim": dim1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]], dtype=np.float32)
    dim2 = 1
    input_dict2 = {"dim": dim2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([0.5, -0.2, 1.7, -0.9], dtype=np.float32)
    dim3 = 0
    input_dict3 = {"dim": dim3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(2, 4, 3).astype(np.float32)
    dim4 = 2
    input_dict4 = {"dim": dim4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1e-6, 1e-5, 1e-4]], dtype=np.float32)
    dim5 = 1
    input_dict5 = {"dim": dim5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    dim6 = 1
    input_dict6 = {"dim": dim6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSoftmax"] = logsoftmax_inputs()

import torch, copy
import numpy as np

def multilabelmarginloss_inputs():
    list_of_inputs = []

    input1 = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    target1 = np.array([[0, 1, -1, 2]], dtype=np.int64)
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
    target2 = np.array([[0, 1, -1, 2], [3, 0, -1, 1]], dtype=np.int64)
    input_dict2 = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-0.1, -0.2, -0.3, -0.4]], dtype=np.float32)
    target3 = np.array([[0, 1, -1, 2]], dtype=np.int64)
    input_dict3 = {
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    target4 = np.array([[0, 1, 2, 3]], dtype=np.int64)
    input_dict4 = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    target5 = np.array([[0, -1], [1, 0]], dtype=np.int64)
    input_dict5 = {
        "size_average": True,
        "reduce": False,
        "reduction": "sum",
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[0.7, 0.8, 0.9]], dtype=np.float32)
    target6 = np.array([[0, 1, 2]], dtype=np.int64)
    input_dict6 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
    target7 = np.array([[1, 3, -1, 0]], dtype=np.int64)
    input_dict7 = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    target8 = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    input_dict8 = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input8,
        "target": target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    target9 = np.array([[3, 2, 1, 0]], dtype=np.int64)
    input_dict9 = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input9,
        "target": target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)
    target10 = np.array([[0, 1, 2]], dtype=np.int64)
    input_dict10 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input10,
        "target": target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.MultiLabelMarginLoss"] = multilabelmarginloss_inputs()

import torch, copy
import numpy as np

def multi_margin_loss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(10, 5).astype(np.float32)
    target1 = np.random.randint(0, 5, size=10).astype(np.int64)
    input_dict1 = {
        "p": 1,
        "margin": 1.0,
        "weight": np.ones(5).astype(np.float32),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(5, 10).astype(np.float32)
    target2 = np.random.randint(0, 10, size=5).astype(np.int64)
    input_dict2 = {
        "p": 2,
        "margin": 0.5,
        "weight": np.random.rand(10).astype(np.float32),
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 2).astype(np.float32)
    target3 = np.array([0, 1]).astype(np.int64)
    input_dict3 = {
        "p": 1,
        "margin": 2.0,
        "weight": np.array([0.5, 0.5]).astype(np.float32),
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(15, 8).astype(np.float32)
    target4 = np.random.randint(0, 8, size=15).astype(np.int64)
    input_dict4 = {
        "p": 2,
        "margin": 1.5,
        "weight": np.ones(8).astype(np.float32),
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.MultiMarginLoss"] = multi_margin_loss_inputs()

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "lower": 0.1,
        "upper": 0.3,
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {
        "lower": 0.2,
        "upper": 0.4,
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 2, 3).numpy()
    input_dict3 = {
        "lower": -0.1,
        "upper": 0.1,
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1).numpy()
    input_dict4 = {
        "lower": 0.0,
        "upper": 0.5,
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 4, 4, 4).numpy()
    input_dict5 = {
        "lower": 0.15,
        "upper": 0.35,
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 2).numpy()
    input_dict6 = {
        "lower": -0.2,
        "upper": -0.1,
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(10).numpy()
    input_dict7 = {
        "lower": 0.125,
        "upper": 0.3333333333333333,
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(3, 3, 3).numpy()
    input_dict8 = {
        "lower": 0.5,
        "upper": 0.6,
        "inplace": True,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(2, 5).numpy()
    input_dict9 = {
        "lower": -0.5,
        "upper": 0.0,
        "inplace": False,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 1, 1, 1).numpy()
    input_dict10 = {
        "lower": 0.05,
        "upper": 0.25,
        "inplace": True,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.RReLU"] = rrelu_inputs()

import torch, copy
import numpy as np

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    input1 = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding1 = 1
    input_dict1 = {"padding": padding1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 3, 4, 4).numpy()
    padding2 = (1, 1, 1, 1)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.zeros(2, 2, 5, 5).numpy()
    padding3 = 1
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.ones(1, 1, 2, 2).numpy()
    padding4 = (0, 0, 0, 0)
    input_dict4 = {"padding": padding4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 5, 3, 3).numpy()
    padding5 = 1
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(3, 1, 2, 2).numpy()
    padding6 = (1, 1, 0, 0)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(1, 1, 5, 5).numpy()
    padding7 = (1, 1, 1, 1)
    input_dict7 = {"padding": padding7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(1, 2, 3, 4).numpy()
    padding8 = 0
    input_dict8 = {"padding": padding8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(2, 3, 3, 3).numpy()
    padding9 = (1, 1, 1, 1)
    input_dict9 = {"padding": padding9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 1, 1, 1).numpy()
    padding10 = 0
    input_dict10 = {"padding": padding10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_1"] = reflectionpad2d_inputs()

import torch, copy
import numpy as np

def reflectionpad2d_inputs():
    list_of_inputs = []

    input1 = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    padding1 = (2, 2, 2, 2)
    input_dict1 = {"padding": padding1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_2"] = reflectionpad2d_inputs()

import torch, copy
import numpy as np

def replicationpad3d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(16, 3, 8, 320, 480).numpy()
    padding1 = 3
    input_dict1 = {"padding": padding1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(1, 5, 10, 20, 30).numpy()
    padding2 = (1, 2, 3, 4, 5, 6)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(4, 2, 5, 10, 15).numpy()
    padding3 = 0
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 1, 3, 7, 11).numpy()
    padding4 = (2, 2, 1, 1, 0, 0)
    input_dict4 = {"padding": padding4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(8, 6, 4, 8, 12).numpy()
    padding5 = -1 
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 1, 1, 1).numpy()
    padding6 = (5, 5, 5, 5, 5, 5)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(32, 1, 16, 64, 128).numpy()
    padding7 = 10
    input_dict7 = {"padding": padding7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 4, 6, 9, 12).numpy()
    padding8 = (1, 1, 2, 2, 3, 3)
    input_dict8 = {"padding": padding8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(5, 2, 3, 4, 5).numpy()
    padding9 = 2
    input_dict9 = {"padding": padding9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 3, 7, 11, 13).numpy()
    padding10 = (0, 0, 0, 0, 0, 0)
    input_dict10 = {"padding": padding10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replicationpad3d_inputs()

import torch, copy
import numpy as np

def replicationpad3d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(16, 3, 8, 320, 480)
    padding1 = (3, 3, 6, 6, 1, 1)
    
    input_dict1 = {
        "input": input1.numpy(),
        "padding": padding1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 5, 10, 20, 30)
    padding2 = (2, 2, 2, 2, 2, 2)
    
    input_dict2 = {
        "input": input2.numpy(),
        "padding": padding2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(32, 1, 16, 64, 128)
    padding3 = (1, 2, 0, 1, 3, 4)
    
    input_dict3 = {
        "input": input3.numpy(),
        "padding": padding3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(8, 6, 4, 8, 16)
    padding4 = (0, 0, 0, 0, 0, 0)

    input_dict4 = {
        "input": input4.numpy(),
        "padding": padding4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 4, 5, 10, 15)
    padding5 = (1, 1, 1, 1, 1, 1)
    
    input_dict5 = {
        "input": input5.numpy(),
        "padding": padding5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replicationpad3d_inputs()

import torch, copy
import numpy as np

def silu_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2).astype(np.float32)
    inplace1 = False
    input_dict1 = {"input": input1, "inplace": inplace1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(3, 4).astype(np.float64)
    inplace2 = True
    input_dict2 = {"input": input2, "inplace": inplace2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([-1.0, 0.0, 1.0]).astype(np.float32)
    inplace3 = False
    input_dict3 = {"input": input3, "inplace": inplace3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(5, 2, 3).astype(np.float16)
    inplace4 = True
    input_dict4 = {"input": input4, "inplace": inplace4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.zeros((1, 1)).astype(np.float32)
    inplace5 = False
    input_dict5 = {"input": input5, "inplace": inplace5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.ones((4,)).astype(np.float64)
    inplace6 = True
    input_dict6 = {"input": input6, "inplace": inplace6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    inplace7 = False
    input_dict7 = {"input": input7, "inplace": inplace7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[-2.0, 3.0], [1.0, -1.0]]).astype(np.float64)
    inplace8 = True
    input_dict8 = {"input": input8, "inplace": inplace8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(10).astype(np.float16)
    inplace9 = False
    input_dict9 = {"input": input9, "inplace": inplace9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.full((3, 3), 2.5).astype(np.float32)
    inplace10 = True
    input_dict10 = {"input": input10, "inplace": inplace10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.SiLU"] = silu_inputs()

import torch, copy
import numpy as np

def smooth_l1_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict1 = {
        "beta": 1.0,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict2 = {
        "beta": 0.5,
        "reduction": "sum",
        "size_average": False,
        "reduce": False,
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    target3 = np.array([1.0, -2.0], dtype=np.float32)
    input_dict3 = {
        "beta": 2.0,
        "reduction": "none",
        "size_average": True,
        "reduce": True,
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0], dtype=np.float32)
    target4 = np.array([2.0], dtype=np.float32)
    input_dict4 = {
        "beta": 0.1,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict5 = {
        "beta": 1.0,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target6 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict6 = {
        "beta": 5.0,
        "reduction": "sum",
        "size_average": False,
        "reduce": True,
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0], dtype=np.float32)
    target7 = np.array([0.0], dtype=np.float32)
    input_dict7 = {
        "beta": 0.0,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    target8 = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict8 = {
        "beta": 10.0,
        "reduction": "none",
        "size_average": False,
        "reduce": False,
        "input": input8,
        "target": target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0], dtype=np.float32)
    target9 = np.array([3.0, 4.0], dtype=np.float32)
    input_dict9 = {
        "beta": 1.5,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input9,
        "target": target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[-1.0], [2.0]], dtype=np.float32)
    target10 = np.array([[-1.0], [2.0]], dtype=np.float32)
    input_dict10 = {
        "beta": 0.5,
        "reduction": "sum",
        "size_average": False,
        "reduce": True,
        "input": input10,
        "target": target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smooth_l1_loss_inputs()

import torch, copy
import numpy as np

def softmax_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, 0.0, 1.0],
                       [2.0, -3.0, 4.0]], dtype=np.float32)
    dim2 = 0
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[0.5, 0.2, 0.3],
                       [0.1, 0.4, 0.5]], dtype=np.float64)
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    dim4 = 0
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    dim5 = 0
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1e-8, 1e-7, 1e-6]], dtype=np.float32)
    dim6 = 0
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-100, -50, 0, 50, 100]], dtype=np.float32)
    dim7 = 0
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    dim8 = 0
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    
    input10 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    dim10 = 0
    input_dict10 = {"input": input10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmax"] = softmax_inputs()

import torch, copy
import numpy as np

def softshrink_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2).numpy()
    lambd1 = 0.5
    input_dict1 = {"lambd": lambd1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 4).numpy()
    lambd2 = 0.1
    input_dict2 = {"lambd": lambd2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 2, 3).numpy()
    lambd3 = 0.8
    input_dict3 = {"lambd": lambd3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    lambd4 = 0.2
    input_dict4 = {"lambd": lambd4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd5 = 0.7
    input_dict5 = {"lambd": lambd5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1.5, -2.5, 3.5]).numpy()
    lambd6 = 2.0
    input_dict6 = {"lambd": lambd6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(1).numpy()
    lambd7 = 0.0
    input_dict7 = {"lambd": lambd7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(4, 1).numpy()
    lambd8 = 1.0
    input_dict8 = {"lambd": lambd8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([np.nan, np.inf, -np.inf]).numpy()
    lambd9 = 0.3
    input_dict9 = {"lambd": lambd9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(2, 2, 2).numpy()
    lambd10 = 0.9
    input_dict10 = {"lambd": lambd10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    input1 = np.array([[-1.0, 0.5, 2.0]])
    target1 = np.array([[0.0, 1.0, 0.0]])
    weight1 = np.array([0.2, 0.5, 0.3])
    size_average1 = True
    reduce1 = True
    pos_weight1 = np.array([1.0])
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "size_average": size_average1,
        "reduce": reduce1,
        "pos_weight": pos_weight1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[0.1, -0.8], [1.2, 0.3]])
    target2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    weight2 = np.array([1.0, 1.0])
    size_average2 = False
    reduce2 = True
    pos_weight2 = np.array([2.0])
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": size_average2,
        "reduce": reduce2,
        "pos_weight": pos_weight2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-2.0, 1.5, -0.7, 3.1]])
    target3 = np.array([[0.0, 1.0, 0.0, 1.0]])
    weight3 = np.array([0.8, 0.2, 0.5, 0.9])
    size_average3 = True
    reduce3 = True
    pos_weight3 = np.array([1.0])
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "size_average": size_average3,
        "reduce": reduce3,
        "pos_weight": pos_weight3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-0.5, 0.2, -1.0, 1.8, 0.7]])
    target4 = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    weight4 = np.array([0.3, 0.7, 0.4, 0.6, 0.5])
    size_average4 = False
    reduce4 = True
    pos_weight4 = np.array([1.0])

    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "size_average": size_average4,
        "reduce": reduce4,
        "pos_weight": pos_weight4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits"] = binary_cross_entropy_with_logits_inputs()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha1 = 1.0
    inplace1 = False
    input_dict1 = {
        "input": input1,
        "alpha": alpha1,
        "inplace": inplace1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    alpha2 = 1.0
    inplace2 = False
    input_dict2 = {
        "input": input2,
        "alpha": alpha2,
        "inplace": inplace2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    alpha3 = 1.0
    inplace3 = False
    input_dict3 = {
        "input": input3,
        "alpha": alpha3,
        "inplace": inplace3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    alpha4 = 1.0
    inplace4 = False
    input_dict4 = {
        "input": input4,
        "alpha": alpha4,
        "inplace": inplace4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha5 = 2.0
    inplace5 = False
    input_dict5 = {
        "input": input5,
        "alpha": alpha5,
        "inplace": inplace5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha6 = 1.0
    inplace6 = True
    input_dict6 = {
        "input": input6,
        "alpha": alpha6,
        "inplace": inplace6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    alpha7 = 2.0
    inplace7 = True
    input_dict7 = {
        "input": input7,
        "alpha": alpha7,
        "inplace": inplace7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    alpha8 = 1.0
    inplace8 = False
    input_dict8 = {
        "input": input8,
        "alpha": alpha8,
        "inplace": inplace8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha9 = 0.5
    inplace9 = False
    input_dict9 = {
        "input": input9,
        "alpha": alpha9,
        "inplace": inplace9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    alpha10 = 1.5
    inplace10 = True
    input_dict10 = {
        "input": input10,
        "alpha": alpha10,
        "inplace": inplace10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

import torch, copy
import numpy as np

def dropout_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict2 = {
        "input": input2,
        "p": 0.2,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "p": 0.8,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([-1.0, -2.0, -3.0])
    input_dict4 = {
        "input": input4,
        "p": 0.1,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.zeros((5, 5))
    input_dict5 = {
        "input": input5,
        "p": 0.9,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.ones((1, 2, 3, 4))
    input_dict6 = {
        "input": input6,
        "p": 0.7,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0, 0.0, 0.0])
    input_dict7 = {
        "input": input7,
        "p": 0.3,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.1, 2.2, 3.3])
    input_dict8 = {
        "input": input8,
        "p": 0.6,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(10)
    input_dict9 = {
        "input": input9,
        "p": 0.4,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1.0], [2.0], [3.0]])
    input_dict10 = {
        "input": input10,
        "p": 0.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []
    
    input1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    lambd1 = 0.5
    input_dict1 = {"input": input1, "lambd": lambd1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([0.2, -0.8, 1.5, -2.1], dtype=np.float32)
    lambd2 = 0.7
    input_dict2 = {"input": input2, "lambd": lambd2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    lambd3 = 0.3
    input_dict3 = {"input": input3, "lambd": lambd3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    lambd4 = 0.9
    input_dict4 = {"input": input4, "lambd": lambd4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-0.9, 0.1, -0.5, 0.7], dtype=np.float32)
    lambd5 = 0.0
    input_dict5 = {"input": input5, "lambd": lambd5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([2.5, -1.2, 0.8, -3.1], dtype=np.float32)
    lambd6 = 1.0
    input_dict6 = {"input": input6, "lambd": lambd6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    lambd7 = 0.25
    input_dict7 = {"input": input7, "lambd": lambd7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lambd8 = 0.6
    input_dict8 = {"input": input8, "lambd": lambd8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.1, -2.2, 3.3], [-4.4, 5.5, -6.6]], dtype=np.float32)
    lambd9 = 0.8
    input_dict9 = {"input": input9, "lambd": lambd9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100.0, -50.0, 25.0], dtype=np.float32)
    lambd10 = 0.4
    input_dict10 = {"input": input10, "lambd": lambd10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

import torch, copy
import numpy as np

def hardswish_inputs():
    list_of_inputs = []
    
    input1 = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float32)
    inplace1 = False
    
    input2 = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    inplace2 = True
    
    input3 = np.array([[-3.0, 3.0], [4.0, -4.0]], dtype=np.float32)
    inplace3 = False
    
    input4 = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    inplace4 = True
    
    input5 = np.array([[-5.0], [0.0], [5.0]], dtype=np.float32)
    inplace5 = False
    
    input6 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    inplace6 = True
    
    input7 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    inplace7 = False

    input8 = np.array([[-6.0, -7.0], [8.0, 9.0]], dtype=np.float32)
    inplace8 = True
    
    input9 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inplace9 = False
    
    input10 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    inplace10 = True
    
    input_dict1 = {"input": input1, "inplace": inplace1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input_dict2 = {"input": input2, "inplace": inplace2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input_dict3 = {"input": input3, "inplace": inplace3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_dict4 = {"input": input4, "inplace": inplace4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input_dict5 = {"input": input5, "inplace": inplace5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input_dict6 = {"input": input6, "inplace": inplace6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input_dict7 = {"input": input7, "inplace": inplace7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input_dict8 = {"input": input8, "inplace": inplace8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input_dict9 = {"input": input9, "inplace": inplace9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {"input": input10, "inplace": inplace10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

import torch, copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    input2 = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    label = np.array([1, 0, 1], dtype=np.float32)
    margin = 0.2
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "label": label,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.margin_ranking_loss"] = margin_ranking_loss_inputs()

import torch, copy
import numpy as np

def pdist_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p1 = 2.0
    
    input_dict1 = {"input": input1, "p": p1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p2 = 1.0
    
    input_dict2 = {"input": input2, "p": p2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    p3 = 2.0
    
    input_dict3 = {"input": input3, "p": p3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    p4 = 3.0
    
    input_dict4 = {"input": input4, "p": p4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0], [2.0], [3.0]])
    p5 = 2.0
    
    input_dict5 = {"input": input5, "p": p5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    p6 = 0.5
    
    input_dict6 = {"input": input6, "p": p6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    p7 = 3.0
    
    input_dict7 = {"input": input7, "p": p7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p8 = np.inf
    
    input_dict8 = {"input": input8, "p": p8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.5, 2.5], [3.5, 4.5]])
    p9 = 1.5
    
    input_dict9 = {"input": input9, "p": p9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[10.0, 20.0], [30.0, 40.0]])
    p10 = 4.0
    
    input_dict10 = {"input": input10, "p": p10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.pdist"] = pdist_inputs()

import torch, copy
import numpy as np

def prelu_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2, 3).numpy()
    weight1 = torch.randn(3).numpy()
    
    input_dict1 = {
        "input": input1,
        "weight": weight1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.functional.prelu"] = prelu_inputs()

import torch, copy
import numpy as np

def relu6_inputs():
    list_of_inputs = []
    
    input1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 7.0], dtype=np.float32)
    inplace1 = False
    input_dict1 = {"input": input1, "inplace": inplace1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    inplace2 = True
    input_dict2 = {"input": input2, "inplace": inplace2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.0, 6.0, 6.1, -1.0], dtype=np.float32)
    inplace3 = False
    input_dict3 = {"input": input3, "inplace": inplace3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    inplace4 = True
    input_dict4 = {"input": input4, "inplace": inplace4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-5.0, 0.0, 3.0, 6.0, 10.0], dtype=np.float32)
    inplace5 = False
    input_dict5 = {"input": input5, "inplace": inplace5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    inplace6 = False
    input_dict6 = {"input": input6, "inplace": inplace6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-2.5, 1.5, 7.2], [0.0, -3.1, 5.8]], dtype=np.float32)
    inplace7 = True
    input_dict7 = {"input": input7, "inplace": inplace7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([4.0], dtype=np.float32)
    inplace8 = False
    input_dict8 = {"input": input8, "inplace": inplace8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    inplace9 = True
    input_dict9 = {"input": input9, "inplace": inplace9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    inplace10 = False
    input_dict10 = {"input": input10, "inplace": inplace10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.relu6"] = relu6_inputs()

import torch, copy
import numpy as np

def selu_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inplace1 = False
    input_dict1 = {"input": input1, "inplace": inplace1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    inplace2 = True
    input_dict2 = {"input": input2, "inplace": inplace2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inplace3 = False
    input_dict3 = {"input": input3, "inplace": inplace3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    inplace4 = True
    input_dict4 = {"input": input4, "inplace": inplace4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    inplace5 = False
    input_dict5 = {"input": input5, "inplace": inplace5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32)
    inplace6 = True
    input_dict6 = {"input": input6, "inplace": inplace6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([], dtype=np.float32)
    inplace7 = False
    input_dict7 = {"input": input7, "inplace": inplace7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1e-5, 1e-6, 1e-7], dtype=np.float32)
    inplace8 = True
    input_dict8 = {"input": input8, "inplace": inplace8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1e5, 1e6, 1e7], dtype=np.float32)
    inplace9 = False
    input_dict9 = {"input": input9, "inplace": inplace9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    inplace10 = True
    input_dict10 = {"input": input10, "inplace": inplace10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

import torch, copy
import numpy as np

def constant_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3], dtype=np.float32)
    val1 = 0.5
    
    input_dict1 = {
        "tensor": input1,
        "val": val1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    val2 = -1.0
    
    input_dict2 = {
        "tensor": input2,
        "val": val2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.zeros((3, 4, 5), dtype=np.float32)
    val3 = 2.71828
    
    input_dict3 = {
        "tensor": input3,
        "val": val3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.ones((2, 2), dtype=np.float16)
    val4 = -0.1
    
    input_dict4 = {
        "tensor": input4,
        "val": val4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(5, 5).astype(np.float32)
    val5 = 10.0
    
    input_dict5 = {
        "tensor": input5,
        "val": val5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0], dtype=np.float64)
    val6 = 0.0
    
    input_dict6 = {
        "tensor": input6,
        "val": val6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.full((1, 2, 3), 3.14, dtype=np.float32)
    val7 = -2.0
    
    input_dict7 = {
        "tensor": input7,
        "val": val7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([], dtype=np.float32).reshape(0, 0)
    val8 = 1.5
    
    input_dict8 = {
        "tensor": input8,
        "val": val8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.arange(10, dtype=np.float32).reshape(2, 5)
    val9 = -5.0
    
    input_dict9 = {
        "tensor": input9,
        "val": val9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.linspace(0, 1, 10, dtype=np.float32).reshape(5, 2)
    val10 = 7.0
    
    input_dict10 = {
        "tensor": input10,
        "val": val10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.init.constant_"] = constant_inputs()

import torch, copy
import numpy as np

def not_equal_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([1, 2, 3]).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, 2, 3]).numpy()
    other2 = torch.tensor([4, 5, 6]).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other3 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other4 = torch.tensor([[5, 6], [7, 8]]).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([True, False, True]).numpy()
    other5 = torch.tensor([True, True, False]).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([-1, 0, 1]).numpy()
    other6 = torch.tensor([1, 0, -1]).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other7 = torch.tensor([1.0, 2.0, 4.0]).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other8 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(2, 3, 4).numpy()
    other9 = torch.randn(2, 3, 4).numpy()
    input_dict9 = {"input": input9, "other": other9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([0]).numpy()
    other10 = torch.tensor([1]).numpy()
    input_dict10 = {"input": input10, "other": other10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.not_equal"] = not_equal_inputs()

import torch, copy
import numpy as np

def numel_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2, 3, 4).numpy()
    list_of_inputs.append({"input": input1})
    
    input2 = torch.zeros(5, 5, 5, 5).numpy()
    list_of_inputs.append({"input": input2})
    
    input3 = torch.ones((1, 2, 3, 4, 5)).numpy()
    list_of_inputs.append({"input": input3})
    
    input4 = torch.randint(0, 10, (2, 2)).numpy()
    list_of_inputs.append({"input": input4})
    
    input5 = torch.randn(3).numpy()
    list_of_inputs.append({"input": input5})
    
    input6 = torch.zeros((10,)).numpy()
    list_of_inputs.append({"input": input6})
    
    input7 = torch.ones((2, 2, 2, 2, 2)).numpy()
    list_of_inputs.append({"input": input7})
    
    input8 = torch.randn(1, 1, 1, 1).numpy()
    list_of_inputs.append({"input": input8})
    
    input9 = torch.randint(0, 100, (4, 4, 4)).numpy()
    list_of_inputs.append({"input": input9})
    
    input10 = torch.zeros((100, 1)).numpy()
    list_of_inputs.append({"input": input10})

    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

import torch, copy
import numpy as np

def permute_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 5).numpy()
    dims1 = (2, 0, 1)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(4, 2).numpy()
    dims2 = (1, 0)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3).numpy()
    dims3 = (0,)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4, 5).numpy()
    dims4 = (3, 1, 0, 2)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5, 4, 3, 2, 1).numpy()
    dims5 = (4, 3, 2, 1, 0)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(10, 10, 10).numpy()
    dims6 = (0, 1, 2)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 2, 2, 2).numpy()
    dims7 = (1, 0, 3, 2)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(7, 8, 9).numpy()
    dims8 = (2, 1, 0)
    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(6, 5).numpy()
    dims9 = (1, 0)
    input_dict9 = {"input": input9, "dims": dims9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 2, 3, 4).numpy()
    dims10 = (3, 2, 1, 0)
    input_dict10 = {"input": input10, "dims": dims10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

import torch, copy
import numpy as np

def positive_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(5).numpy()
    input_dict = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input2 = torch.randn(2, 3).numpy()
    input_dict = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input3 = torch.randn(3, 4, 5).numpy()
    input_dict = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input4 = torch.tensor([1.0, -2.0, 3.0]).numpy()
    input_dict = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input5 = torch.tensor([[1.0, 2.0], [-3.0, 4.0]]).numpy()
    input_dict = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input6 = torch.zeros(5).numpy()
    input_dict = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input7 = torch.ones(2, 2).numpy()
    input_dict = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input8 = torch.randn(1).numpy()
    input_dict = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input9 = torch.tensor([0.0]).numpy()
    input_dict = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input10 = torch.randn(4,1).numpy()
    input_dict = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

import torch, copy
import numpy as np

def reshape_inputs():
    list_of_inputs = []

    input1 = torch.arange(12.).numpy()
    shape1 = (3, 4)
    input_dict1 = {"input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[0, 1], [2, 3]]).numpy()
    shape2 = (-1,)
    input_dict2 = {"input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4).numpy()
    shape3 = (6, 4)
    input_dict3 = {"input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1, 2, 3, 4, 5, 6]).numpy()
    shape4 = (2, -1)
    input_dict4 = {"input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randint(0, 10, (5,)).numpy()
    shape5 = (1, 5)
    input_dict5 = {"input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.zeros(2, 2, 2).numpy()
    shape6 = (4,2)
    input_dict6 = {"input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.ones((1, 3, 2)).numpy()
    shape7 = (3, 2)
    input_dict7 = {"input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.rand(4, 4).numpy()
    shape8 = (16,)
    input_dict8 = {"input": input8, "shape": shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    shape9 = (2, 2)
    input_dict9 = {"input": input9, "shape": shape9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

import torch, copy
import numpy as np

def round_inputs():
    list_of_inputs = []

    input1 = np.array([4.7, -2.3, 9.1, -7.7])
    decimals1 = 0
    out1 = np.empty_like(input1)

    input_dict1 = {
        "input": input1,
        "decimals": decimals1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-0.5, 0.5, 1.5, 2.5])
    decimals2 = 0
    out2 = np.empty_like(input2)

    input_dict2 = {
        "input": input2,
        "decimals": decimals2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.1234567])
    decimals3 = 3
    out3 = np.empty_like(input3)

    input_dict3 = {
        "input": input3,
        "decimals": decimals3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1200.1234567])
    decimals4 = -3
    out4 = np.empty_like(input4)

    input_dict4 = {
        "input": input4,
        "decimals": decimals4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(2, 2)
    decimals5 = 2
    out5 = np.empty_like(input5)

    input_dict5 = {
        "input": input5,
        "decimals": decimals5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0])
    decimals6 = 0
    out6 = np.empty_like(input6)

    input_dict6 = {
        "input": input6,
        "decimals": decimals6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([-1.0, -2.0, -3.0])
    decimals7 = 1
    out7 = np.empty_like(input7)

    input_dict7 = {
        "input": input7,
        "decimals": decimals7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 4, 5)
    decimals8 = -2
    out8 = np.empty_like(input8)

    input_dict8 = {
        "input": input8,
        "decimals": decimals8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([10000.0], dtype=np.float16)
    decimals9 = 3
    out9 = np.empty_like(input9)

    input_dict9 = {
        "input": input9,
        "decimals": decimals9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([123.456])
    decimals10 = -1
    out10 = np.empty_like(input10)

    input_dict10 = {
        "input": input10,
        "decimals": decimals10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.round"] = round_inputs()

import torch, copy
import numpy as np

def set_num_interop_threads_inputs():
    list_of_inputs = []
    
    input1 = np.int32(1)
    input_dict1 = {"num_threads": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.set_num_interop_threads"] = set_num_interop_threads_inputs()

import torch, copy
import numpy as np

def signbit_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.7, -1.2, 0., 2.3], dtype=np.float32)
    out1 = np.empty_like(input1, dtype=np.bool_)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-0.0, 0.0], dtype=np.float32)
    out2 = np.empty_like(input2, dtype=np.bool_)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    out3 = np.empty_like(input3, dtype=np.bool_)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    out4 = np.empty_like(input4, dtype=np.bool_)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
    out5 = np.empty_like(input5, dtype=np.bool_)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[-0.0, 0.0], [0.0, -0.0]], dtype=np.float64)
    out6 = np.empty_like(input6, dtype=np.bool_)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, -1.0, 2.0, -2.0, 0.0], dtype=np.float32)
    out7 = np.empty_like(input7, dtype=np.bool_)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[-1.5, 0.0, 2.7], [-3.1, -0.0, 4.2]], dtype=np.float64)
    out8 = np.empty_like(input8, dtype=np.bool_)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1e-9, -1e-9, 0.0], dtype=np.float32)
    out9 = np.empty_like(input9, dtype=np.bool_)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[-1e10, 1e10], [-1e-10, 1e-10]], dtype=np.float64)
    out10 = np.empty_like(input10, dtype=np.bool_)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.signbit"] = signbit_inputs()

import torch, copy
import numpy as np

def torch_sin_inputs():
    list_of_inputs = []

    input1 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    out1 = np.zeros_like(input1)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-np.pi/2, -np.pi, -3*np.pi/2])
    out2 = np.zeros_like(input2)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]])
    out3 = np.zeros_like(input3)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-np.pi/2, -np.pi], [-3*np.pi/2, -2*np.pi]])
    out4 = np.zeros_like(input4)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0])
    out5 = np.zeros_like(input5)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([-1.0, -2.0, -3.0])
    out6 = np.zeros_like(input6)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[0.0, np.pi/2], [np.pi, 3*np.pi/2]], [[-np.pi/2, -np.pi], [-3*np.pi/2, -2*np.pi]]])
    out7 = np.zeros_like(input7)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(5, 5)
    out8 = np.zeros_like(input8)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.randn(2, 3, 4)
    out9 = np.zeros_like(input9)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.0])
    out10 = np.zeros_like(input10)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.sin"] = torch_sin_inputs()

import torch, copy
import numpy as np

def i1e_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0]).numpy()
    list_of_inputs.append({"x": input1})
    
    input2 = torch.tensor([-1.0, 2.0]).numpy()
    list_of_inputs.append({"x": input2})
    
    input3 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    list_of_inputs.append({"x": input3})
    
    input4 = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    list_of_inputs.append({"x": input4})
    
    input5 = torch.tensor([0.0]).numpy()
    list_of_inputs.append({"x": input5})
    
    input6 = torch.tensor([1e-9]).numpy()
    list_of_inputs.append({"x": input6})

    input7 = torch.tensor([1e9]).numpy()
    list_of_inputs.append({"x": input7})

    input8 = torch.tensor([1.0, 2.0]).numpy()
    list_of_inputs.append({"x": input8})
    
    input9 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    list_of_inputs.append({"x": input9})
    
    input10 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    list_of_inputs.append({"x": input10})
    
    input11 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).numpy()
    list_of_inputs.append({"x": input11})

    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

import torch, copy
import numpy as np

def polygamma_inputs():
    list_of_inputs = []
    
    input1 = np.array(0, dtype=np.int64)
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    input_dict1 = {
        "n": input1,
        "x": torch.tensor(x1).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array(1, dtype=np.int64)
    x2 = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    
    input_dict2 = {
        "n": input2,
        "x": torch.tensor(x2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array(2, dtype=np.int64)
    x3 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    
    input_dict3 = {
        "n": input3,
        "x": torch.tensor(x3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array(3, dtype=np.int64)
    x4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    
    input_dict4 = {
        "n": input4,
        "x": torch.tensor(x4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array(0, dtype=np.int64)
    x5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    
    input_dict5 = {
        "n": input5,
        "x": torch.tensor(x5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array(1, dtype=np.int64)
    x6 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    
    input_dict6 = {
        "n": input6,
        "x": torch.tensor(x6).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    
    input7 = np.array(4, dtype=np.int64)
    x7 = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    
    input_dict7 = {
        "n": input7,
        "x": torch.tensor(x7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array(0, dtype=np.int64)
    x8 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    
    input_dict8 = {
        "n": input8,
        "x": torch.tensor(x8).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.special.polygamma"] = polygamma_inputs()

import torch, copy, numpy as np

def sinc_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.0, 1.0, 2.0, 3.0]).astype(np.float64)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]).astype(np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[0.0, 1.0], [2.0, 3.0]]).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, -2.0], [-3.0, -4.0]]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0]).astype(np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([-1.0]).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([0.0, 0.0, 0.0]).astype(np.float64)
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([10.0, 20.0, 30.0]).astype(np.float32)
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(5, 5).astype(np.float64)
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.special.sinc"] = sinc_inputs()

import torch, copy
import numpy as np

def torch_sqrt_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 4.0, 9.0, 16.0])
    out1 = np.array([])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([0.0, 0.25, 0.5, 0.75])
    out2 = np.array([])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-1.0, 4.0, -9.0, 16.0])
    out3 = np.array([])
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 3)
    out4 = np.array([])
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0])
    out5 = np.array([])
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([2.0, 3.0, 5.0])
    out6 = np.array([])
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([16.0, 25.0, 36.0, 49.0])
    out7 = np.array([])
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 2)
    out8 = np.array([])
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(3, 2, 2)
    out9 = np.array([])
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.0, 1.0, 2.0, 3.0])
    out10 = np.array([])
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.sqrt"] = torch_sqrt_inputs()

import torch, copy
import numpy as np

def take_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    index1 = torch.tensor([0, 2]).numpy()
    input_dict1 = {"input": input1, "index": index1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([10, 20, 30, 40, 50]).numpy()
    index2 = torch.tensor([1, 3, 4]).numpy()
    input_dict2 = {"input": input2, "index": index2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(2, 3, 4).numpy()
    index3 = torch.tensor([0, 3, 6, 9]).numpy()
    input_dict3 = {"input": input3, "index": index3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    index4 = torch.tensor([0, 4, 8]).numpy()
    input_dict4 = {"input": input4, "index": index4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    index5 = torch.tensor([-1, 1, 3]).numpy()
    input_dict5 = {"input": input5, "index": index5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(5, 5).numpy()
    index6 = torch.tensor([0, 0, 0, 0, 0]).numpy()
    input_dict6 = {"input": input6, "index": index6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([[1, 2], [3, 4]]).numpy()
    index7 = torch.tensor([1, 1]).numpy()
    input_dict7 = {"input": input7, "index": index7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([1, 2, 3]).numpy()
    index8 = torch.tensor([0, 1, 2]).numpy()
    input_dict8 = {"input": input8, "index": index8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(2, 2).numpy()
    index9 = torch.tensor([3]).numpy()
    input_dict9 = {"input": input9, "index": index9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([10, 20, 30]).numpy()
    index10 = torch.tensor([0, 0, 0]).numpy()
    input_dict10 = {"input": input10, "index": index10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.take"] = take_inputs()

import torch, copy
import numpy as np

def tril_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(3, 3).astype(np.float32)
    diagonal1 = 0
    out1 = np.zeros((3, 3)).astype(np.float32)
    input_dict1 = {"input": input1, "diagonal": diagonal1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(4, 6).astype(np.float64)
    diagonal2 = 1
    out2 = np.zeros((4, 6)).astype(np.float64)
    input_dict2 = {"input": input2, "diagonal": diagonal2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 2).astype(np.float16)
    diagonal3 = -1
    out3 = np.zeros((2, 2)).astype(np.float16)
    input_dict3 = {"input": input3, "diagonal": diagonal3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(5, 5).astype(np.complex128)
    diagonal4 = 2
    out4 = np.zeros((5, 5)).astype(np.complex128)
    input_dict4 = {"input": input4, "diagonal": diagonal4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(1, 1).astype(np.float32)
    diagonal5 = 0
    out5 = np.zeros((1, 1)).astype(np.float32)
    input_dict5 = {"input": input5, "diagonal": diagonal5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 3).astype(np.float64)
    diagonal6 = -2
    out6 = np.zeros((2, 3)).astype(np.float64)
    input_dict6 = {"input": input6, "diagonal": diagonal6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(3, 2).astype(np.float32)
    diagonal7 = 1
    out7 = np.zeros((3, 2)).astype(np.float32)
    input_dict7 = {"input": input7, "diagonal": diagonal7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(4, 4).astype(np.complex64)
    diagonal8 = -1
    out8 = np.zeros((4, 4)).astype(np.complex64)
    input_dict8 = {"input": input8, "diagonal": diagonal8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(5, 5).astype(np.float16)
    diagonal9 = 3
    out9 = np.zeros((5, 5)).astype(np.float16)
    input_dict9 = {"input": input9, "diagonal": diagonal9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 2).astype(np.float64)
    diagonal10 = -2
    out10 = np.zeros((2, 2)).astype(np.float64)
    input_dict10 = {"input": input10, "diagonal": diagonal10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.tril"] = tril_inputs()

import torch, copy
import numpy as np

def unique_consecutive_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 1, 2, 2, 3, 4, 4, 4, 5])
    input_dict1 = {
        "input": input1,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([1, 2, 3, 4, 5])
    input_dict2 = {
        "input": input2,
        "return_inverse": False,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([])
    input_dict3 = {
        "input": input3,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 1, 1, 1, 1])
    input_dict4 = {
        "input": input4,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([5, 4, 3, 2, 1])
    input_dict5 = {
        "input": input5,
        "return_inverse": False,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 1, 2, 1, 2])
    input_dict6 = {
        "input": input6,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, -1, 1, -1, 1])
    input_dict7 = {
        "input": input7,
        "return_inverse": True,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1, 1], [2, 2], [3, 3]])
    input_dict8 = {
        "input": input8,
        "return_inverse": False,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1, 2], [1, 2], [3, 4]])
    input_dict9 = {
        "input": input9,
        "return_inverse": True,
        "return_counts": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    input_dict10 = {
        "input": input10,
        "return_inverse": True,
        "return_counts": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.unique_consecutive"] = unique_consecutive_inputs()

