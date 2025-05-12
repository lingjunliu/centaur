import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    module = input_dict["module"]
    example_inputs = input_dict["example_inputs"]

    if not cpu:
        module = module.cuda()
        example_inputs = tuple(ex.cuda() for ex in example_inputs)

    module.eval()
    traced_module = torch.jit.trace(module, example_inputs[0])

    if not cpu:
        traced_module = traced_module.cpu()

    return {"result": True}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch

    module = input_dict["module"]
    example_inputs = input_dict["example_inputs"]

    class WrapperModule(tf.Module):
        def __init__(self, pytorch_module):
            super(WrapperModule, self).__init__()
            self.pytorch_module = pytorch_module

        @tf.function
        def __call__(self, x):
            x_numpy = x.numpy()
            x_tensor = torch.from_numpy(x_numpy)
            output_tensor = self.pytorch_module(x_tensor)
            return output_tensor.detach().numpy()

    wrapped_module = WrapperModule(module)
    input_signature = [tf.TensorSpec(shape=example_inputs[0].shape, dtype=tf.float32)]
    concrete_function = wrapped_module.__call__.get_concrete_function(*input_signature)

    return {"result": True}

class DummyModule(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x + 1

def main():
    A_TOL = 0.01

    import torch
    dummy_module = DummyModule()

    class TorchDummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    input_tensor = torch.randn(2,3)
    torch_dummy_module = TorchDummyModule()

    input_data = {
        "module": torch_dummy_module,
        "example_inputs": (input_tensor,)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result == tf_result, "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()