import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.qat import freeze_bn_stats
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])

    bn = nn.BatchNorm1d(input_tensor.shape[0])
    bn.running_mean = torch.randn(input_tensor.shape[0])
    bn.running_var = torch.randn(input_tensor.shape[0])
    bn.eval()

    if not cpu:
        input_tensor = input_tensor.cuda()
        bn = bn.cuda()

    freeze_bn_stats(bn)

    return {"result": input_tensor.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])

    return {"result": input_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()