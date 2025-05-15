import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    init = input_dict.get("init", 0.25)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.PReLU(num_parameters=input_tensor.size(1) if len(input_tensor.shape) > 1 else 1, init=init)
    if not cpu:
        m = m.cuda()
    result = m(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        init = input_dict.get("init", 0.25)

        num_parameters = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
        alpha = tf.Variable(tf.fill([num_parameters], init), dtype=tf.float32)

        if len(input_tensor.shape) > 1:
            alpha_reshaped = alpha
        else:
            alpha_reshaped = alpha

        result = tf.maximum(input_tensor, 0) + alpha_reshaped * tf.minimum(input_tensor, 0)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[-0.4544, -0.2471,  0.7741, -0.5713,  0.2641],
                           [ 0.8362, -0.0355,  1.4702, -0.8775,  0.1783]], dtype=np.float32),
        "init": 0.3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()