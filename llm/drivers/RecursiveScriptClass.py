import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    @torch.jit.script
    class MyModule(object):
        def __init__(self, n):
            self.n = n

        def forward(self, x):
            return x + self.n

    my_module = MyModule(input_tensor)

    result = my_module.forward(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"])
        
        class MyModule(tf.Module):
            def __init__(self, n):
                self.n = tf.constant(n, dtype=input_tensor.dtype)

            @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
            def forward(self, x):
                return x + self.n

        my_module = MyModule(input_tensor.numpy())
        result = my_module.forward(input_tensor)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()