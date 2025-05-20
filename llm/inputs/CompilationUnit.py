
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def CompilationUnit_inputs():
    list_of_inputs = []

    # Input 1: Simple function definition
    input1 = """
    def foo(x):
        return x + 1
    """
    list_of_inputs.append({"source": input1, "input": torch.randn(1).numpy()})

    # Input 2: Function with type hints
    input2 = """
    def bar(x: int) -> int:
        return x * 2
    """
    list_of_inputs.append({"source": input2, "input": torch.randint(0, 10, (1,)).numpy()})

    # Input 3: Class definition with a method
    input3 = """
    class MyClass:
        def __init__(self, value: float):
            self.value = value

        def get_value(self) -> float:
            return self.value
    """
    list_of_inputs.append({"source": input3, "input": torch.randn(1, 1).numpy()})

    # Input 4: Multiple functions and classes
    input4 = """
    def add(x: int, y: int) -> int:
        return x + y

    class Point:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

        def distance_from_origin(self) -> float:
            return (self.x**2 + self.y**2)**0.5
    """
    list_of_inputs.append({"source": input4, "input": torch.randn(2, 2).numpy()})

    # Input 5: A function that uses torch
    input5 = """
    import torch

    def create_tensor(size: int) -> torch.Tensor:
        return torch.randn(size)
    """
    list_of_inputs.append({"source": input5, "input": torch.randn(3, 3).numpy()})


    return list_of_inputs

generated_inputs = CompilationUnit_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CompilationUnit', generated_inputs)
