import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.DoubleStorage(input_tensor.size()[0])
    result.copy_(input_tensor.storage())

    if not cpu:
        result = torch.DoubleStorage.cpu(result)

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float64)

    if not cpu:
        with tf.device("/GPU:0"):
            tf_result = tf.Variable(tf.zeros(input_tensor.shape, dtype=tf.float64))
            tf_result.assign(input_tensor)
            result = tf_result.numpy()

    else:
        tf_result = tf.Variable(tf.zeros(input_tensor.shape, dtype=tf.float64))
        tf_result.assign(input_tensor)
        result = tf_result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()