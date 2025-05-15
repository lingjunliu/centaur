import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", "tanh")
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)

    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    class DummyRNN(torch.nn.RNNBase):
        def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, bidirectional, device):
            mode = "ReLU" if nonlinearity == 'relu' else "Tanh" if nonlinearity == 'tanh' else "RNN"
            super(DummyRNN, self).__init__(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                mode=mode
            )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.nonlinearity = nonlinearity
            self.bias = bias
            self.batch_first = batch_first
            self.dropout = dropout
            self.bidirectional = bidirectional
            self.device = device

        def forward(self, input, hx=None):
            batch_size = input.size(0) if self.batch_first else input.size(1)
            num_directions = 2 if self.bidirectional else 1
            if hx is None:
                hx = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device)
            return input, hx

    rnn = DummyRNN(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, bidirectional, device)

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    hx = input_dict.get("hx")
    if hx is not None:
      hx = torch.tensor(hx, dtype=torch.float32)
    
    if not cpu:
        rnn = rnn.cuda()
        input_tensor = input_tensor.cuda()
        if hx is not None:
          hx = hx.cuda()

    output, hn = rnn(input_tensor, hx)

    if not cpu:
        output = output.cpu()
        hn = hn.cpu()

    return {"output": output.detach().numpy(), "hn": hn.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    from tensorflow.keras.layers import SimpleRNN, RNN, Dense, Activation
    from tensorflow.keras import initializers

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", "tanh")
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hx = input_dict.get("hx")
        if hx is not None:
          hx = tf.constant(hx, dtype=tf.float32)

        if batch_first:
          input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        def get_initializer(seed=42):
            return initializers.GlorotUniform(seed=seed)
        
        rnn_layers = []

        if nonlinearity == "relu":
            activation = tf.nn.relu
        else:
            activation = tf.nn.tanh

        for i in range(num_layers):
            cell = SimpleRNN(hidden_size, activation=activation, use_bias=bias, kernel_initializer=get_initializer(), recurrent_initializer=get_initializer(), return_sequences=True if i < num_layers - 1 else False, return_state=True)
            rnn_layers.append(cell)
          
        output = input_tensor
        initial_state = tf.split(hx, num_layers, axis=0) if hx is not None else None

        states = []
        for i, rnn in enumerate(rnn_layers):
            if initial_state:
                output, state = rnn(output, initial_state=initial_state[i])
            else:
                output, state = rnn(output)
            states.append(state)

        hn = tf.stack(states)

        if bidirectional:
            hn = tf.reshape(hn, (num_layers * 2, tf.shape(hn)[1], hidden_size))

        if batch_first:
          output = tf.transpose(output, perm=[1, 0, 2])
          
        output = output.numpy()
        hn = hn.numpy()

    return {"output": output, "hn": hn}

def main():
    A_TOL = 0.01

    input_data = {
        "input_size": 10,
        "hidden_size": 20,
        "input": np.random.randn(5, 3, 10).astype(np.float32),
        "num_layers": 2,
        "bidirectional": False,
        "batch_first": True,
        "hx": np.random.randn(2, 3, 20).astype(np.float32),
        "nonlinearity": "relu"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Output results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Hidden state results do not match"

    print("Success")

if __name__ == "__main__":
    main()