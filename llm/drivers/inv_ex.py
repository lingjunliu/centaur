import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.linalg.inv_ex(input_tensor)

    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result": result[0].numpy(), "info": result[1].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        try:
            inverse = tf.linalg.inv(input_tensor)
            info = tf.constant(0, dtype=tf.int32)
        except tf.errors.InvalidArgumentError:
            rank = len(input_tensor.shape)
            if rank == 0:
                raise ValueError("input should have at least rank 1")
            elif rank == 1:
                info = tf.constant(-10, dtype=tf.int32)
            elif rank >= 2:
                try:
                    s = tf.linalg.svd(input_tensor)

                    if isinstance(s, tuple):
                        singular_values = s[0]
                    else:
                        singular_values = s
                        
                    min_singular_value = singular_values[..., -1]
                    max_singular_value = singular_values[..., 0]

                    rcond = 1e-15
                    tol = tf.maximum(rcond * max_singular_value, tf.cast(0.0, max_singular_value.dtype))
                    rank = tf.reduce_sum(tf.cast(singular_values > tol, tf.int32))

                    if rank < tf.shape(input_tensor)[-1]:
                        info = tf.constant(-11, dtype=tf.int32)
                    else:
                        info = tf.constant(0, dtype=tf.int32)
                except Exception as e:
                    info = tf.constant(-11, dtype=tf.int32)

            inverse = tf.zeros_like(input_tensor)
        inverse = inverse.numpy()
        info = info.numpy()

    return {"result": inverse, "info": info}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info do not match"
    
    input_data = {
        "input": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1.0, 0.0], [0.0, 1e-16]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Results do not match"

    
    print("Success")

if __name__ == "__main__":
    main()