import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict['input'])
    hx = (torch.tensor(input_dict['hx']), torch.tensor(input_dict['cx']))
    i_weights = torch.tensor(input_dict['i_weights'])
    h_weights = torch.tensor(input_dict['h_weights'])
    i_biases = torch.tensor(input_dict['i_biases'])
    h_biases = torch.tensor(input_dict['h_biases'])
    use_bias = input_dict.get('use_bias', True)
    train = input_dict.get('training', False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = (hx[0].cuda(), hx[1].cuda())
        i_weights = i_weights.cuda()
        h_weights = h_weights.cuda()
        i_biases = i_biases.cuda()
        h_biases = h_biases.cuda()

    lstm = nn.LSTM(input_size=input_tensor.shape[2], hidden_size=h_weights.shape[1], num_layers=hx[0].shape[0], batch_first=True)
    lstm.weight_ih_l0 = nn.Parameter(i_weights)
    lstm.weight_hh_l0 = nn.Parameter(h_weights)
    lstm.bias_ih_l0 = nn.Parameter(i_biases)
    lstm.bias_hh_l0 = nn.Parameter(h_biases)
    
    output, (h_n, c_n) = lstm(input_tensor, hx)

    if not cpu:
        output = output.cpu()
        h_n = h_n.cpu()
        c_n = c_n.cpu()

    return {'result_0': output.detach().numpy(), 'result_1': h_n.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict['input'], dtype=tf.float32)
        h_weights = tf.constant(input_dict['h_weights'], dtype=tf.float32)
        i_weights = tf.constant(input_dict['i_weights'], dtype=tf.float32)
        h_biases = tf.constant(input_dict['h_biases'], dtype=tf.float32)
        i_biases = tf.constant(input_dict['i_biases'], dtype=tf.float32)
        hx = tf.constant(input_dict['hx'], dtype=tf.float32)
        cx = tf.constant(input_dict['cx'], dtype=tf.float32)
        use_bias = input_dict.get('use_bias', True)
        training = input_dict.get('training', False)

        lstm_cell = tf.keras.layers.LSTMCell(units=h_weights.shape[1], 
                                              kernel_initializer=tf.keras.initializers.constant(i_weights.numpy()),
                                              recurrent_initializer=tf.keras.initializers.constant(h_weights.numpy()),
                                              bias_initializer=tf.keras.initializers.constant(np.concatenate([i_biases.numpy(), h_biases.numpy()])),
                                              use_bias=use_bias,
                                              )

        lstm = tf.keras.layers.RNN(cell=lstm_cell, return_sequences=True, return_state=True)
        output, h, c = lstm(input_tensor, initial_state=[hx, cx])
    
        return {'result_0': output.numpy(), 'result_1': h.numpy()}

def main():
    A_TOL = 0.01
    hidden_size = 20
    input_size = 10
    batch_size = 5
    seq_len = 3
    num_layers = 1

    input_data = {
        'input': np.random.rand(batch_size, seq_len, input_size).astype(np.float32),
        'h_weights': np.random.rand(hidden_size, hidden_size).astype(np.float32),
        'i_weights': np.random.rand(input_size, hidden_size).astype(np.float32),
        'h_biases': np.random.rand(hidden_size).astype(np.float32),
        'i_biases': np.random.rand(hidden_size).astype(np.float32),
        'hx': np.random.rand(num_layers, batch_size, hidden_size).astype(np.float32),
        'cx': np.random.rand(num_layers, batch_size, hidden_size).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result['result_0'], tf_result['result_0'], atol=A_TOL)
    assert np.allclose(torch_result['result_1'], tf_result['result_1'], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()