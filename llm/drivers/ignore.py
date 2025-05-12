import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    class MyModule(nn.Module):
        def __init__(self, drop):
            super().__init__()
            self.training = True
            self.drop = drop

        @torch.jit.ignore(drop=False)
        def training_method(self, x):
            return

        def forward(self, x):
            if self.training:
                if not self.drop:
                    self.training_method(x)
            return x

    x = torch.tensor(input_dict["x"])
    if not cpu:
        x = x.cuda()
    
    module = MyModule(input_dict.get("drop", False))

    if not cpu:
        module = module.cuda()

    m = torch.jit.script(module)
    result = m(x)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    class MyModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.training = True

        def training_method(self, x):
            pass

        @tf.function
        def forward(self, x):
            if self.training:
                if input_dict.get("drop", False):
                    tf.print("training_method was dropped")
                    return x
                else:
                    self.training_method(x)
            return x

    x = tf.constant(input_dict["x"])
    if not cpu:
        with tf.device("/GPU:0"):
            module = MyModule()
            result = module.forward(x)
            result = result.numpy()
    else:
        module = MyModule()
        result = module.forward(x).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "drop": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "drop": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()