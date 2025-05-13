import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    tau = torch.tensor(input_dict["tau"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        tau = tau.cuda()
    
    result = torch.linalg.householder_product(input_tensor, tau)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        tau = tf.constant(input_dict["tau"], dtype=tf.float32)

        m = input_tensor.shape[0]
        n = input_tensor.shape[1]
        k = tau.shape[0]

        Q = tf.identity(input_tensor)
        for i in range(k):
            v = tf.concat([tf.zeros([i], dtype=input_tensor.dtype),
                             tf.ones([1], dtype=input_tensor.dtype),
                             tf.reshape(input_tensor[i+1:, i], [-1])], axis=0)

            H = tf.eye(m, dtype=input_tensor.dtype) - tau[i] * tf.expand_dims(v, 1) @ tf.expand_dims(v, 0)

            Q = tf.matmul(Q, H)

        result = Q.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "tau": np.array([0.5, 0.6, 0.7], dtype=np.float32)
    }
    
    input_data["input"] = np.eye(3, dtype=np.float32)
    input_data["tau"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()