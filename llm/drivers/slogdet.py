import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    
    if not cpu:
        A = A.cuda()
    
    sign, logabsdet = torch.linalg.slogdet(A)
    
    if not cpu:
        sign = sign.cpu()
        logabsdet = logabsdet.cpu()
    
    return {"sign": sign.numpy(), "logabsdet": logabsdet.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        
        det = tf.linalg.det(A)
        sign = tf.sign(det)
        logabsdet = tf.math.log(tf.abs(det))
        
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
    
    return {"sign": sign, "logabsdet": logabsdet}

def main():
    A_TOL = 0.01
    input_data = {
        "A": np.array([[0.0032, -0.2239, -1.1219], 
                       [-0.6690, 0.1161, 0.4053], 
                       [-1.6218, -0.9273, -0.0082]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["sign"], tf_result["sign"], atol=A_TOL), "Sign results do not match"
    assert np.allclose(torch_result["logabsdet"], tf_result["logabsdet"], atol=A_TOL), "Logabsdet results do not match"

    print("Success")

if __name__ == "__main__":
    main()