import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    bias = input_dict.get("bias", True)
    num_chunks = input_dict.get("num_chunks", 1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        
    class MyRNNCellBase(torch.nn.RNNCellBase):
        def __init__(self, input_size, hidden_size, bias, num_chunks):
            super(MyRNNCellBase, self).__init__(input_size, hidden_size, bias, num_chunks)

        def forward(self, input, hx):
            weight_ih = self.weight_ih
            weight_hh = self.weight_hh
            bias_ih = self.bias_ih
            bias_hh = self.bias_hh
            
            intermediate = torch.matmul(input, weight_ih.t()) + torch.matmul(hx, weight_hh.t())
            if bias_ih is not None and bias_hh is not None:
                intermediate += bias_ih + bias_hh

            return torch.tanh(intermediate)

    cell = MyRNNCellBase(input_size, hidden_size, bias, num_chunks)
    
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
    bias_hh = torch.tensor(input_dict["bias_hh"]) if "bias_hh" in input_dict else None
    
    cell.weight_ih = torch.nn.Parameter(weight_ih)
    cell.weight_hh = torch.nn.Parameter(weight_hh)
    if bias_ih is not None:
        cell.bias_ih = torch.nn.Parameter(bias_ih)
        cell.bias_hh = torch.nn.Parameter(bias_hh)
    else:
        cell.bias_ih = None
        cell.bias_hh = None

    hx = torch.tensor(input_dict["hx"])

    if not cpu:
        hx = hx.cuda()
        cell.cuda()
        
    result = cell(input_tensor, hx)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]
        bias = input_dict.get("bias", True)

        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
        bias_hh = tf.constant(input_dict["bias_hh"]) if "bias_hh" in input_dict else None
        
        hx = tf.constant(input_dict["hx"])

        intermediate = tf.matmul(input_tensor, weight_ih, transpose_b=True) + tf.matmul(hx, weight_hh, transpose_b=True)
        if bias:
            intermediate += bias_ih + bias_hh

        result = tf.tanh(intermediate)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_size = 3
    hidden_size = 5
    batch_size = 2

    input_data = {
        "input": np.random.rand(batch_size, input_size).astype(np.float32),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "weight_ih": np.random.rand(hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.rand(hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.rand(hidden_size).astype(np.float32),
        "bias_hh": np.random.rand(hidden_size).astype(np.float32),
        "hx": np.random.rand(batch_size, hidden_size).astype(np.float32),
        "bias": True,
        "num_chunks": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()