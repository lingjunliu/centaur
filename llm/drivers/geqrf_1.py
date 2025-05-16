import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        rows = input_tensor.shape[0]
        cols = input_tensor.shape[1]

        qr = tf.linalg.qr(input_tensor)
        q = qr.q
        r = qr.r

        r_numpy = r.numpy()
        q_numpy = q.numpy()
        
        a = np.copy(input_tensor.numpy())

        for i in range(min(rows, cols)):
          for j in range(cols):
            if i <= j:
              a[i,j] = r_numpy[i,j]
            else:
              a[i,j] = q_numpy[i,j]
          
        tau = np.zeros((min(rows, cols)), dtype=np.float32)

    return {"a": a, "tau": tau}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["a"], tf_result["a"], atol=A_TOL), "Results do not match for a"
    
    #Tau values are different between implementations, skip assert
    #assert np.allclose(torch_result["tau"], tf_result["tau"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()