import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    correction = input_dict.get("correction", 1)
    fweights = input_dict.get("fweights", None)
    aweights = input_dict.get("aweights", None)

    if fweights is not None:
        fweights = torch.tensor(fweights)
    if aweights is not None:
        aweights = torch.tensor(aweights)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if fweights is not None:
            fweights = fweights.cuda()
        if aweights is not None:
            aweights = aweights.cuda()

    result = torch.cov(input_tensor, correction=correction, fweights=fweights, aweights=aweights)

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
        correction = input_dict.get("correction", 1)
        fweights = input_dict.get("fweights", None)
        aweights = input_dict.get("aweights", None)

        if fweights is not None:
            fweights = tf.constant(fweights, dtype=tf.float32)
        if aweights is not None:
            aweights = tf.constant(aweights, dtype=tf.float32)

        input_shape = input_tensor.shape
        if len(input_shape) <= 1:
            input_tensor = tf.reshape(input_tensor, (1, -1)) if len(input_shape) == 1 else tf.reshape(input_tensor, (1, 1))
        
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)

        num_rows = tf.cast(tf.shape(input_tensor)[0], dtype=tf.float32)
        num_cols = tf.cast(tf.shape(input_tensor)[1], dtype=tf.float32)

        if fweights is None and aweights is None:
            means = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
            centered_data = input_tensor - means
            covariance_matrix = tf.matmul(tf.transpose(centered_data), centered_data) / (num_cols - correction)

        else:
            if fweights is None:
                w = aweights
            elif aweights is None:
                w = fweights
            else:
                w = fweights * aweights

            w = tf.cast(w, dtype=tf.float32)
            sum_w = tf.reduce_sum(w)
            
            means = tf.reduce_sum(input_tensor * w, axis=1) / sum_w
            means = tf.reshape(means, (-1, 1))
            centered_data = input_tensor - means

            weighted_covariance_matrix = tf.matmul(tf.transpose(centered_data), centered_data * tf.reshape(w, (1,-1))) / (sum_w - (tf.reduce_sum(w*w) / sum_w)*correction)

            covariance_matrix = tf.transpose(weighted_covariance_matrix)
        
        if len(input_shape) <= 1:
            result = tf.reshape(covariance_matrix, ()).numpy()
        else:
            result = covariance_matrix.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_2d = {
        "input": np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32).T,
        "correction": 1
    }

    torch_result = torch_version(input_data_2d)
    tf_result = tensorflow_version(input_data_2d)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_2d = {
        "input": np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32).T,
        "correction": 0
    }

    torch_result = torch_version(input_data_2d)
    tf_result = tensorflow_version(input_data_2d)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_2d = {
        "input": np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32).T,
        "fweights": np.array([1, 6, 9], dtype=np.int32),
        "aweights": np.array([0.4282, 0.0255, 0.4144], dtype=np.float32)
    }

    torch_result = torch_version(input_data_2d)
    tf_result = tensorflow_version(input_data_2d)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_1d = {
        "input": np.array([1,2,3,4,5], dtype=np.float32),
        "correction": 1
    }
    torch_result = torch_version(input_data_1d)
    tf_result = tensorflow_version(input_data_1d)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()