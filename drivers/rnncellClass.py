import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    set_seed()
    # input
    x1 = input["input_size"]
    x2 = input["hidden_size"]
    x3 = torch.tensor(input["input"])
    x4 = torch.tensor(input["hidden"])

    # output
    func_obj = torch.nn.RNNCell(x1, x2)
    
    if not cpu:
        x3 = x3.cuda()
        x4 = x4.cuda()
        func_obj = func_obj.cuda()
    
    y = func_obj(x3, x4)
    
    if not cpu:
        y = y.cpu()

    return {"rnn": y.detach().numpy()}


def tensorflow_version(input, cpu=True):
    set_seed()
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = input["hidden_size"]
        x2 = tf.constant(input["input"])
        x3 = tf.constant(input["hidden"])
        
        # output
        func_obj = tf.keras.layers.SimpleRNNCell(units=x1)
        y, _ = func_obj(x2, [x3])
        
        return {"rnn": y.numpy()}
    

def main():
    # Example input
    input_data = {
        "input": torch.randn(3, 10),
        "hidden": torch.randn(3, 20),
        "input_size": 10,
        "hidden_size": 20
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Compare results
    torch_output = torch_result["rnn"]
    tf_output = tf_result["rnn"]
    
    if np.allclose(torch_output, tf_output, atol=1e-2):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()