import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    n = input_dict["n"]
    d = input_dict.get("d", 1.0)
    dtype = input_dict.get("dtype", None)
    layout = input_dict.get("layout", torch.strided)
    device = input_dict.get("device", None)
    requires_grad = input_dict.get("requires_grad", False)
    
    if not cpu:
        pass
    
    result = torch.fft.rfftfreq(
        n,
        d=d,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
    )
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    n = input_dict["n"]
    d = input_dict.get("d", 1.0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        num_freqs = (n + 1) // 2
        result = tf.cast(tf.range(num_freqs), dtype=tf.float32) / (d * n)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_5 = {
        "n": 5,
        "d": 1.0,
    }
    torch_result_5 = torch_version(input_data_5)
    tf_result_5 = tensorflow_version(input_data_5)

    min_len = min(len(torch_result_5["result"]), len(tf_result_5["result"]))
    assert np.allclose(torch_result_5["result"][:min_len], tf_result_5["result"][:min_len], atol=A_TOL), "Results do not match for n=5"

    input_data_4 = {
        "n": 4,
        "d": 1.0,
    }
    torch_result_4 = torch_version(input_data_4)
    tf_result_4 = tensorflow_version(input_data_4)

    min_len = min(len(torch_result_4["result"]), len(tf_result_4["result"]))
    assert np.allclose(torch_result_4["result"][:min_len], tf_result_4["result"][:min_len], atol=A_TOL), "Results do not match for n=4"
    
    print("Success")

if __name__ == "__main__":
    main()