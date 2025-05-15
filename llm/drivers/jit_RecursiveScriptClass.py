import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    class MyModule(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
    
        def forward(self, x):
            if torch.sum(x) > self.value:
                return x - 1
            else:
                return x + 1
    
    module = MyModule(input_dict.get("value", 0))
    
    if not cpu:
        module = module.cuda()
    
    scripted_module = torch.jit.script(module)
    result = scripted_module(input_tensor)
    
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
        input_tensor = tf.constant(input_dict["input"])
        value = input_dict.get("value", 0)

        def forward(x, value):
            def true_fn():
                return x - 1

            def false_fn():
                return x + 1

            return tf.cond(tf.greater(tf.reduce_sum(x), tf.cast(value, x.dtype)), true_fn, false_fn)

        result = forward(input_tensor, value)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "value": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()