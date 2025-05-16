import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    example_inputs = tuple(torch.tensor(i) for i in input_dict["example_inputs"])
    check = input_dict.get("check", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        example_inputs = tuple(i.cuda() for i in example_inputs)

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        @torch.jit.script_method
        def forward(self, x):
            return x

    module = torch.jit.script(MyModule())
    result = module(input_tensor)


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
        example_inputs = tuple(tf.constant(i) for i in input_dict["example_inputs"])
        check = input_dict.get("check", False)

        def dummy_script_method(input_tensor):
            return input_tensor
        
        result = dummy_script_method(input_tensor)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "example_inputs": [np.array([4.0, 5.0, 6.0], dtype=np.float32)],
        "check": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()