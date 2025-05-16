import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.corrcoef(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)

        if len(input_tensor.shape) == 0:
            result = tf.constant(1.0, dtype=tf.float32)
        elif len(input_tensor.shape) == 1:
            if len(input_tensor) == 1:
                result = tf.constant(1.0, dtype=tf.float32)
            else:
                input_tensor = tf.expand_dims(input_tensor, axis=0)
                mean = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
                centered_data = input_tensor - mean
                covariance_matrix = tf.matmul(centered_data, centered_data, transpose_a=False, transpose_b=True) / tf.cast((tf.shape(input_tensor)[1] - 1), dtype=tf.float32)
                
                std_devs = tf.sqrt(tf.linalg.diag_part(covariance_matrix))
                
                if tf.reduce_any(tf.equal(std_devs, 0.0)):
                  result = tf.eye(tf.shape(covariance_matrix)[0], dtype=tf.float32)
                else:
                  result = covariance_matrix / tf.matmul(tf.expand_dims(std_devs, 1), tf.expand_dims(std_devs, 0))

        else:
            mean = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
            centered_data = input_tensor - mean
            covariance_matrix = tf.matmul(centered_data, centered_data, transpose_b=True) / tf.cast((tf.shape(input_tensor)[1] - 1), dtype=tf.float32)

            std_devs = tf.sqrt(tf.linalg.diag_part(covariance_matrix))
            if tf.reduce_any(tf.equal(std_devs, 0.0)):
              result = tf.eye(tf.shape(covariance_matrix)[0], dtype=tf.float32)
            else:
              result = covariance_matrix / tf.matmul(tf.expand_dims(std_devs, 1), tf.expand_dims(std_devs, 0))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0, 1, 2], [2, 1, 0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.randn(2, 4).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-0.2678, -0.0908, -0.3766, 0.2780], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()