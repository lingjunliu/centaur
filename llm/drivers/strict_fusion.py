import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    x = torch.tensor(input_dict["x"])

    if not cpu:
        x = x.cuda()

    @torch.jit.script
    def foo(x):
        with torch.jit.strict_fusion():
            return x + x + x

    result = foo(x)

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
        x = tf.constant(input_dict["x"])

        result = x + x + x
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()