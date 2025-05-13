import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_linear = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1])
    with torch.no_grad():
        lazy_linear.weight.zero_()
        lazy_linear.bias.zero_()

    result = lazy_linear(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        num_features = input_tensor.shape[-1]
        kernel = tf.Variable(tf.zeros([num_features, num_features]))
        bias = tf.Variable(tf.zeros([num_features]))

        result = tf.matmul(input_tensor, kernel) + bias

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056],
                           [0.0303, 1.1985, 1.4506, -0.7056]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()