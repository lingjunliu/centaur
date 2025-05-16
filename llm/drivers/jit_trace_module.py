import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    mod = input_dict["mod"]
    example_inputs = input_dict["example_inputs"]
    check_trace = input_dict.get("check_trace", True)
    check_inputs = input_dict.get("check_inputs", None)
    strict = input_dict.get("strict", True)

    example_inputs = tuple(torch.tensor(x) for x in example_inputs)
    
    if not cpu:
        for i in range(len(example_inputs)):
          example_inputs[i] = example_inputs[i].cuda()
    
    method_name = "forward"

    traced_module = torch.jit.trace_module(mod, {"forward": example_inputs[0]}, check_trace=check_trace, check_inputs=check_inputs, strict=strict)
    
    if not cpu:
        traced_module = traced_module.cpu()
    
    return {"result": traced_module}

class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

class TFWrapper(tf.Module):
    def __init__(self, model):
        super(TFWrapper, self).__init__()
        self.model = model

        self.vars = []
        for name, param in model.named_parameters():
            self.vars.append(tf.Variable(param.detach().numpy(), name=name, trainable=False))

    @tf.function
    def __call__(self, x):
        x = tf.cast(x, dtype=tf.float32)

        params = list(self.model.parameters())
        for i, var in enumerate(self.vars):
            params[i].data = torch.tensor(var.numpy())

        x = torch.tensor(x.numpy())
        return self.model(x).detach().numpy()

def tensorflow_version(input_dict, cpu=True):
    mod = input_dict["mod"]
    example_inputs = input_dict["example_inputs"]
    check_trace = input_dict.get("check_trace", True)
    check_inputs = input_dict.get("check_inputs", None)
    strict = input_dict.get("strict", True)

    example_inputs = tuple(np.array(x) for x in example_inputs)
    tf_module = TFWrapper(mod)

    for inp in example_inputs:
        tf_module(inp)

    return {"result": tf_module}

def main():
    A_TOL = 0.01
    
    mod = DummyModule()
    example_inputs = [np.random.rand(1, 10).astype(np.float32)]
    input_dict = {
        "mod": mod,
        "example_inputs": example_inputs,
        "check_trace": True,
        "check_inputs": None,
        "strict": True
    }
    
    torch_result = torch_version(input_dict)
    tf_result = tensorflow_version(input_dict)
    
    input_tensor = torch.tensor(input_dict["example_inputs"][0])
    torch_output = torch_result["result"](input_tensor)
    tf_output = tf.convert_to_tensor(tf_result["result"](input_dict["example_inputs"][0]), dtype=tf.float32)
    
    assert np.allclose(torch_output.detach().numpy(), tf_output.numpy(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()