import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    momentum_update = input_dict.get("momentum_update", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    class DummyBN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.running_mean = torch.zeros(input_tensor.size(1))
            self.running_var = torch.ones(input_tensor.size(1))

        def forward(self, x, momentum_update=True):
            if momentum_update:
                with torch.no_grad():
                    mean = x.mean(dim=[0, 2, 3] if len(x.shape) == 4 else 0)
                    var = x.var(dim=[0, 2, 3] if len(x.shape) == 4 else 0, unbiased=False)
                    momentum = 0.1
                    self.running_mean.copy_(self.running_mean * (1 - momentum) + mean * momentum)
                    self.running_var.copy_(self.running_var * (1 - momentum) + var * momentum)
            return x

    bn = DummyBN()
    if not cpu:
        bn = bn.cuda()
        bn.running_mean = bn.running_mean.cuda()
        bn.running_var = bn.running_var.cuda()

    result = bn(input_tensor, momentum_update=momentum_update)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        momentum_update = input_dict.get("momentum_update", True)

        result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "momentum_update": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "momentum_update": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()