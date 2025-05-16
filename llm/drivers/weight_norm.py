import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()


def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"], requires_grad=True)
    name = input_dict.get("name", "weight")
    dim = input_dict.get("dim", 0)

    class MyModule(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, x):
            return self.weight

    module = MyModule(input_tensor)
    if not cpu:
        module = module.cuda()

    torch.nn.utils.weight_norm(module, name=name, dim=dim)

    if not cpu:
        module.weight_v = module.weight_v.cpu()
        module.weight_g = module.weight_g.cpu()
    v = module.weight_v.detach().cpu().numpy()
    g = module.weight_g.detach().cpu().numpy()

    del module
    return {"result_v": v, "result_g": g}


def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        name = input_dict.get("name", "weight")
        dim = input_dict.get("dim", 0)

        shape = input_tensor.shape
        if len(shape) == 0:
            shape = (1,)
        num_rows = 1
        for i in range(dim):
            num_rows *= shape[i]

        num_cols = 1
        for i in range(dim, len(shape)):
            num_cols *= shape[i]

        W = tf.reshape(input_tensor, [num_rows, num_cols])

        norm = tf.sqrt(tf.reduce_sum(tf.square(W), axis=0, keepdims=True) + 1e-8)
        V = W / norm
        G = norm

        V = tf.reshape(V, shape)

        result_v = V.numpy()
        result_g = G.numpy()

    return {"result_v": result_v, "result_g": result_g}


def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "dim": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(
        torch_result["result_v"], tf_result["result_v"], atol=A_TOL
    ), "Results do not match for V"
    assert np.allclose(
        torch_result["result_g"], tf_result["result_g"], atol=A_TOL
    ), "Results do not match for G"

    print("Success")


if __name__ == "__main__":
    main()