import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

def torch_version(input_dict, cpu=True):
    W = torch.tensor(input_dict["W"])
    V = torch.tensor(input_dict["V"])
    g = torch.tensor(input_dict["g"])

    if not cpu:
        W = W.cuda()
        V = V.cuda()
        g = g.cuda()

    class DummyModule(nn.Module):
        def __init__(self, W):
            super().__init__()
            self.W = nn.Parameter(W)

        def forward(self, x):
            return self.W

    dummy_module = DummyModule(W)
    weight_norm(dummy_module, name='W', dim=1)
    
    with torch.no_grad():
      dummy_module.W_g.copy_(g)
      dummy_module.W_v.copy_(V)

    result = dummy_module.forward(W)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    W = tf.constant(input_dict["W"], dtype=tf.float32)
    V = tf.constant(input_dict["V"], dtype=tf.float32)
    g = tf.constant(input_dict["g"], dtype=tf.float32)

    norm_V = tf.linalg.norm(V, axis=-1, keepdims=True)
    W_hat = V / norm_V * g

    return {"result": W_hat.numpy()}


def main():
    A_TOL = 0.01
    input_data = {
        "W": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "V": np.array([[0.5, 0.8], [0.1, 0.2]], dtype=np.float32),
        "g": np.array([[1.0], [2.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()