import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    strategy_list = input_dict["strategy_list"]
    
    if not cpu:
        pass
    
    torch.jit.set_fusion_strategy(strategy_list)
    
    if not cpu:
        pass
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    return {}

def main():
    A_TOL = 0.01
    input_data = {
        "strategy_list": [("STATIC", 0)]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    print("Success")

if __name__ == "__main__":
    main()