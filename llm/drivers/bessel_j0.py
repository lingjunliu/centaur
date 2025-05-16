import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.special.bessel_j0(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.math import cos, sin, sqrt, abs

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        x = tf.constant(input_dict["input"], dtype=tf.float32)
        
        def bessel_j0(x):
            def chebyshev_bessel_j0(x_val):
                x_val_sq = x_val**2
                p = -0.00351635 + x_val_sq * (0.000304292 + x_val_sq * 0.000034674)
                p = -0.000375223 + x_val_sq * (0.0417321 - x_val_sq * 0.000085507)
                p =  0.00262486 + x_val_sq * (0.0135213 - x_val_sq * 0.00030288)
                p = -0.0160341 + x_val_sq * (0.0449773 - x_val_sq * 0.00114915)
                p =  0.0534373 + x_val_sq * (0.0928104 - x_val_sq * 0.00408506)
                p = -0.135757 + x_val_sq * (0.207337 - x_val_sq * 0.0102659)
                p =  0.400441 + x_val_sq * (0.441519 - x_val_sq * 0.047334)
                p =  0.999999 + x_val_sq * (0.0747235 - x_val_sq * 0.160223)
                return p

            def asymptotics_bessel_j0(x_val):
                inverse_x = 1.0 / x_val
                p = 0.00556843 + inverse_x * (0.00126062 + inverse_x * 0.000193654)
                p = 0.0400697 + inverse_x * (0.00696536 + inverse_x * 0.000731493)
                p = 0.316233 + inverse_x * (0.027062 + inverse_x * 0.00404652)
                p = 0.999999 + inverse_x * (0.0903369 + inverse_x * 0.0200844)
                scaling = sqrt(0.63661977236758134308) / sqrt(x_val)
                phase = x_val - 0.78539816339744827899
                return scaling * (p * cos(phase) - inverse_x * p * sin(phase))

            abs_x = abs(x)
            
            results_list = []
            for i in tf.range(tf.size(abs_x)):
                val = abs_x[i]
                if val > 8.0:
                    result = asymptotics_bessel_j0(val)
                else:
                    result = chebyshev_bessel_j0(val)
                results_list.append(result)
            
            return tf.stack(results_list)
        
        result = bessel_j0(x)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()