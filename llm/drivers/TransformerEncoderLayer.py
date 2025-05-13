import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict['input'])
    
    d_model = input_dict.get('d_model')
    nhead = input_dict.get('nhead')
    dim_feedforward = input_dict.get('dim_feedforward', 2048)
    dropout = input_dict.get('dropout', 0.1)
    activation = input_dict.get('activation', 'relu')
    layer_norm_eps = input_dict.get('layer_norm_eps', 1e-5)
    batch_first = input_dict.get('batch_first', False)
    norm_first = input_dict.get('norm_first', False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first)
    
    result = encoder_layer(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {'result': result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict['input'], dtype=tf.float32)

        d_model = input_dict.get('d_model')
        nhead = input_dict.get('nhead')
        dim_feedforward = input_dict.get('dim_feedforward', 2048)
        dropout = input_dict.get('dropout', 0.1)
        activation = input_dict.get('activation', 'relu')
        layer_norm_eps = input_dict.get('layer_norm_eps', 1e-5)
        batch_first = input_dict.get('batch_first', False)
        norm_first = input_dict.get('norm_first', False)

        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'gelu':
            activation_fn = tf.nn.gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        class MultiHeadAttention(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dropout_rate):
                super(MultiHeadAttention, self).__init__()
                self.num_heads = num_heads
                self.d_model = d_model

                assert d_model % self.num_heads == 0

                self.depth = d_model // self.num_heads

                self.wq = tf.keras.layers.Dense(d_model)
                self.wk = tf.keras.layers.Dense(d_model)
                self.wv = tf.keras.layers.Dense(d_model)
                self.dense = tf.keras.layers.Dense(d_model)
                self.dropout = tf.keras.layers.Dropout(dropout_rate)

            def split_heads(self, x, batch_size):
                x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
                return tf.transpose(x, perm=[0, 2, 1, 3])

            def call(self, q, k, v):
                batch_size = tf.shape(q)[0]

                q = self.wq(q)
                k = self.wk(k)
                v = self.wv(v)

                q = self.split_heads(q, batch_size)
                k = self.split_heads(k, batch_size)
                v = self.split_heads(v, batch_size)

                matmul_qk = tf.matmul(q, k, transpose_b=True)
                dk = tf.cast(tf.shape(k)[-1], tf.float32)
                scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

                attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
                output = tf.matmul(attention_weights, v)

                output = tf.transpose(output, perm=[0, 2, 1, 3])
                output = tf.reshape(output, (batch_size, -1, self.d_model))
                output = self.dense(output)
                output = self.dropout(output)

                return output, attention_weights

        class FeedForwardNetwork(tf.keras.layers.Layer):
            def __init__(self, d_model, dff, activation, dropout_rate):
                super(FeedForwardNetwork, self).__init__()
                self.dense1 = tf.keras.layers.Dense(dff, activation=activation)
                self.dense2 = tf.keras.layers.Dense(d_model)
                self.dropout = tf.keras.layers.Dropout(dropout_rate)

            def call(self, x):
                x = self.dense1(x)
                x = self.dropout(x)
                x = self.dense2(x)
                return x

        class LayerNormalization(tf.keras.layers.Layer):
            def __init__(self, epsilon=layer_norm_eps):
                super(LayerNormalization, self).__init__()
                self.epsilon = epsilon

            def build(self, input_shape):
                self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=tf.ones_initializer(), trainable=True)
                self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=tf.zeros_initializer(), trainable=True)

            def call(self, x):
                mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
                variance = tf.math.reduce_variance(x, axis=-1, keepdims=True)
                normalized_x = (x - mean) / tf.math.sqrt(variance + self.epsilon)
                return self.gamma * normalized_x + self.beta

        class TransformerEncoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, rate=0.1, activation='relu', layer_norm_epsilon=1e-5, norm_first=False):
                super(TransformerEncoderLayer, self).__init__()
                self.norm_first = norm_first
                self.mha = MultiHeadAttention(d_model, num_heads, rate)
                self.ffn = FeedForwardNetwork(d_model, dff, activation, rate)
                self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon)
                self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon)
                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)

            def call(self, x):
                if self.norm_first:
                    x_norm = self.layernorm1(x)
                    attn_output, _ = self.mha(x_norm, x_norm, x_norm)
                    x = x + self.dropout1(attn_output)

                    x_norm = self.layernorm2(x)
                    ffn_output = self.ffn(x_norm)
                    x = x + self.dropout2(ffn_output)
                else:
                    attn_output, _ = self.mha(x, x, x)
                    x = self.layernorm1(x + self.dropout1(attn_output))

                    ffn_output = self.ffn(x)
                    x = self.layernorm2(x + self.dropout2(ffn_output))
                return x

        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first)
        result = encoder_layer(input_tensor)

        if batch_first:
            result = tf.transpose(result, perm=[1, 0, 2])

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.random.rand(32, 10, 512).astype(np.float32),
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-5,
        'batch_first': False,
        'norm_first': False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()