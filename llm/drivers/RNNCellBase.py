import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]

    class DummyRNNCellBase(torch.nn.modules.RNNCellBase):
        def __init__(self, input_size, hidden_size, bias=True, num_chunks=1):
            super(DummyRNNCellBase, self).__init__(input_size, hidden_size, bias, num_chunks)

        def forward(self, input, hx):
            return torch.add(input, hx)

    cell = DummyRNNCellBase(input_size, hidden_size)

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        cell = cell.cuda()

    result = cell(input_tensor, hx)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]

    class DummyRNNCellBase:
        def __init__(self, input_size, hidden_size):
            self.input_size = input_size
            self.hidden_size = hidden_size

        def __call__(self, input_tensor, hx):
            return tf.add(input_tensor, hx)

    cell = DummyRNNCellBase(input_size, hidden_size)
    input_tensor = tf.constant(input_dict["input"])
    hx = tf.constant(input_dict["hx"])

    result = cell(input_tensor, hx)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input_size": 5,
        "hidden_size": 5,
        "input": np.random.rand(1, 5).astype(np.float32),
        "hx": np.random.rand(1, 5).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()