import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):

    torch.manual_seed(0)

    init = input_dict["init"]

    if not cpu:
        pass

    result = init(device='cpu').forward()
    
    if not cpu:
        pass
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):

    tf.random.set_seed(0)

    init = input_dict["init"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        result = init()
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    class SimpleModel(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(3, 4, device=device))
            self.b = torch.nn.Parameter(torch.randn(2, 2, device=device))

        def forward(self):
            return torch.zeros(3, 4, device=self.a.device)
    
    def tf_simple_model():
        return tf.zeros((3,4),dtype=tf.float32)
    

    input_data = {
        "init": SimpleModel
    }

    torch_result = torch_version(input_data)
    input_data_tf = {
        "init": tf_simple_model
    }
    tf_result = tensorflow_version(input_data_tf)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()