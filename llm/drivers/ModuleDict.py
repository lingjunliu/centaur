import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    modules = input_dict["modules"]
    
    if not cpu:
        for key, module in modules.items():
          modules[key] = module.cuda()
    
    result = torch.nn.ModuleDict(modules)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": dict((key, module.state_dict()) for (key,module) in result.items())}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch
    modules = input_dict["modules"]
    result_dict = {}
    for key, module in modules.items():
      weights = module.state_dict()
      tf_weights = {}
      for name, param in weights.items():
        tf_name = name.replace('.', '_')
        tf_weights[tf_name] = tf.constant(param.numpy())
      result_dict[key] = tf_weights
    return {"result": result_dict}

def main():
    A_TOL = 0.01
    import torch.nn as nn
    import torch
    # Example input
    input_data = {
        "modules": {
          'linear': nn.Linear(20, 30),
          'conv': nn.Conv2d(3, 6, 5),
        }
    }
    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    assert len(torch_result["result"]) == len(tf_result["result"]), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()