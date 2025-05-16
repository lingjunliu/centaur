import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch import Tensor

    def foo(a: Tensor, b: int) -> Tensor:
        return a + b

    def bar(a):
        fut: torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)
        return torch.jit.wait(fut)

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    script_bar = torch.jit.script(bar)

    result = script_bar(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    def foo(a, b):
        return a + tf.cast(b, dtype=a.dtype)

    @tf.function
    def bar(a):
        return foo(a, 2)

    input_tensor = tf.constant(input_dict["input"])

    if not cpu:
        with tf.device('/GPU:0'):
            result = bar(input_tensor).numpy()
    else:
        with tf.device('/CPU:0'):
            result = bar(input_tensor).numpy()
    
    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()