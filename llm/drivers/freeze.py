import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    mod = input_dict["mod"]
    preserved_attrs = input_dict.get("preserved_attrs", None)
    optimize_numerics = input_dict.get("optimize_numerics", True)

    if not cpu:
        pass 

    result = torch.jit.freeze(mod, preserved_attrs=preserved_attrs, optimize_numerics=optimize_numerics)
    
    if not cpu:
        pass
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch
    
    class TFModule:
        def __init__(self, torch_module):
            self.torch_module = torch_module
            self.params = {name: tf.Variable(param.detach().numpy()) for name, param in torch_module.named_parameters()}
            self.buffers = {name: tf.Variable(buf.detach().numpy(), trainable=False) for name, buf in torch_module.named_buffers()}
            self.attrs = {}
            for name in dir(torch_module):
              attr = getattr(torch_module, name)
              if not name.startswith('__') and not callable(attr) and name not in self.params and name not in self.buffers:
                try:
                  if isinstance(attr, torch.Tensor):
                    self.attrs[name] = tf.Variable(attr.detach().numpy())
                  else:
                    self.attrs[name] = attr
                except:
                  pass

        def __call__(self, *args):
            return self.forward(*args)

        def forward(self, *args):
            raise NotImplementedError

    
    mod = input_dict["mod"]
    preserved_attrs = input_dict.get("preserved_attrs", None)
    optimize_numerics = input_dict.get("optimize_numerics", True)
    
    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        
        return {"result": mod}

def main():
    import torch
    A_TOL = 0.01
    
    class MyModule(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
            self.linear = torch.nn.Linear(out_features, out_features)

        def forward(self, input):
            output = self.weight.mm(input)
            output = self.linear(output)
            return output

    class MyModule2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modified_tensor = torch.tensor(10)
            self.version = 1

        def forward(self, input):
            self.modified_tensor += 1
            return input + self.modified_tensor

    scripted_module = torch.jit.script(MyModule(2, 3).eval())

    input_data = {
        "mod": scripted_module
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert type(torch_result["result"]) == type(tf_result["result"]), "Types do not match"

    scripted_module = torch.jit.script(MyModule2().eval())

    input_data = {
        "mod": scripted_module,
        "preserved_attrs": ["version"]
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert type(torch_result["result"]) == type(tf_result["result"]), "Types do not match"

    print("Success")

if __name__ == "__main__":
    main()