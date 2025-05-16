import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.special.modified_bessel_k1(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.math import exp, log, lgamma, pow, sqrt, special

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        x = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)

        def bessel_k1(x):

            def log_modified_bessel_k1(x):
                near = x <= 3.75
                far = x > 3.75
                
                log_bk1_near = tf.where(near,
                    tf.math.log(0.5 / x) - x + tf.add_n([
                        -0.421977348881E-01 * tf.ones_like(x),
                        -0.143572356129E-01 * x,
                        0.247079637586E-02 * pow(x, 2),
                        -0.234863301400E-03 * pow(x, 3),
                        0.104450559690E-04 * pow(x, 4),
                        -0.134075774078E-05 * pow(x, 5),
                        0.333283419170E-07 * pow(x, 6),
                    ]), tf.zeros_like(x,dtype=tf.float32))
                
                log_bk1_far = tf.where(far,
                    -x + tf.math.log(sqrt(tf.constant(3.141592653589793, dtype=tf.float32) / (2 * x))) +
                    tf.add_n([
                        0.398942280401E+00 * tf.ones_like(x),
                        0.305145453174E-01 / x,
                        -0.916232471115E-02 / pow(x, 2),
                        0.205770652868E-02 / pow(x, 3),
                        -0.263079743617E-03 / pow(x, 4),
                        0.163557982973E-04 / pow(x, 5),
                        -0.708532172886E-05 / pow(x, 6),
                        0.100343476694E-05 / pow(x, 7),
                    ]), tf.zeros_like(x,dtype=tf.float32))
                
                return tf.where(x > 0, tf.where(near, log_bk1_near, log_bk1_far), -100*tf.ones_like(x,dtype=tf.float32))

            return tf.math.exp(log_modified_bessel_k1(x))

        result = bessel_k1(x)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.2

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, 2.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()