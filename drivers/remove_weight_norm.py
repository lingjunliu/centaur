import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    module = input_dict["module"]
    name = input_dict["name"]
    
    if not cpu:
        module = module.cuda()
    
    try:
        torch.nn.utils.remove_weight_norm(module, name)
    except ValueError:
        print(f"Warning: weight_norm of '{name}' not found in {module}")
    
    if not cpu:
        module = module.cpu()
    
    return {"result": None}

def tensorflow_version(input_dict, cpu=True):
    module = input_dict["module"]
    name = input_dict["name"]
    
    try:
        layer = getattr(module, name)

        if hasattr(layer, 'kernel'):
            weight_v = layer.kernel
            weight = tf.Variable(weight_v)
            layer.kernel = weight
            
        if hasattr(module, name + '_g'):
            delattr(module, name + '_g')
        if hasattr(module, name + '_v'):
            delattr(module, name + '_v')
        delattr(module, name)
        setattr(module, name, layer) 
    except AttributeError:
        print(f"Warning: Attribute '{name}' not found in {module}")
    
    return {"result": None}

class DummyModuleTorch(torch.nn.Module):
    def __init__(self):
        super(DummyModuleTorch, self).__init__()
        self.linear = torch.nn.Linear(10, 5)
        torch.nn.utils.weight_norm(self.linear, dim=0)

class DummyModuleTF(tf.Module):
    def __init__(self):
        super(DummyModuleTF, self).__init__()
        self.linear = tf.keras.layers.Dense(5, input_dim=10)
        #self.linear.build(input_shape=(None, 10))
        #self.linear.kernel = tf.Variable(self.linear.kernel)
        #self.linear.bias = tf.Variable(self.linear.bias)
        
        self._name = 'DummyModuleTF'
    

def main():
    A_TOL = 0.01

    module = DummyModuleTorch()

    input_data = {
        "module": module,
        "name": "linear"
    }

    torch_result = torch_version(input_data, cpu=True)

    
    class WrappedTFModule:
        def __init__(self, module):
            self.linear = module.linear
    module_tf = DummyModuleTF()
    module_tf_wrapped = WrappedTFModule(module_tf)
        
    input_data_tf = {
        "module": module_tf_wrapped,
        "name": "linear"
    }
    tensorflow_result = tensorflow_version(input_data_tf, cpu=True)

    print("Success")

if __name__ == "__main__":
    main()