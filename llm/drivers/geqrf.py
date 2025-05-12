import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    a, tau = torch.geqrf(input_tensor)

    if not cpu:
        a = a.cpu()
        tau = tau.cpu()
    
    return {"a": a.numpy(), "tau": tau.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        
        q, r = tf.linalg.qr(input_tensor)
        
        m, n = input_tensor.shape
        k = min(m, n)
        
        a = np.copy(input_tensor.numpy())
        a = np.tril(a, -1) + np.triu(r.numpy())
        
        return {"a": a, "tau": np.zeros((k,), dtype=np.float32)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[12.0, -51.0, 4.0],
                           [6.0, 167.0, -68.0],
                           [-4.0, 24.0, -41.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["a"], tf_result["a"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["tau"], tf_result["tau"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()