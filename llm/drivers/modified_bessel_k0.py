import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.special.modified_bessel_k0(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        x = tf.constant(input_dict["input"], dtype=tf.float32)

        def modified_bessel_k0(x):
            x = tf.cast(x, tf.float64)
            result = tf.where(
                tf.less(x, 3.75),
                -tf.math.log(x / 2.0) * tf.math.bessel_i0(x) - 0.57721566 + (x / 2.0) ** 2 * (
                    0.42278420 + (x / 2.0) ** 2 * (0.23069756 + (x / 2.0) ** 2 * (0.03488590 + (x / 2.0) ** 2 * (0.00262698)))),
                tf.exp(-x) / tf.sqrt(x) * (1.25331414 + 0.01234258 / x - 0.00036556 / (x * x) + 0.00002100 / (x * x * x))
            )
            return tf.cast(result, tf.float32)

        result = modified_bessel_k0(x)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, 2.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()