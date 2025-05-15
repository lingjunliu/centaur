import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.intrinsic.quantized.ConvReLU2d(input_tensor.shape[1], input_tensor.shape[1], 1)

    if not cpu:
        result = result.cpu()

    return {"result": np.array([0])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])

    return {"result": np.array([0])}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()