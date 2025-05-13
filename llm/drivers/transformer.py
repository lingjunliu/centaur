import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict['input'])
    d_model = input_dict.get('d_model', 512)
    nhead = input_dict.get('nhead', 8)
    num_encoder_layers = input_dict.get('num_encoder_layers', 6)
    num_decoder_layers = input_dict.get('num_decoder_layers', 6)
    dim_feedforward = input_dict.get('dim_feedforward', 2048)
    dropout = input_dict.get('dropout', 0.1)
    activation = input_dict.get('activation', 'relu')
    custom_encoder = input_dict.get('custom_encoder', None)
    custom_decoder = input_dict.get('custom_decoder', None)
    layer_norm_eps = input_dict.get('layer_norm_eps', 1e-5)
    batch_first = input_dict.get('batch_first', False)
    norm_first = input_dict.get('norm_first', False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    transformer_model = torch.nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        custom_encoder=custom_encoder,
        custom_decoder=custom_decoder,
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first,
        norm_first=norm_first
    )

    if not cpu:
        transformer_model = transformer_model.cuda()

    src = input_tensor
    tgt = input_tensor

    result = transformer_model(src, tgt)

    if not cpu:
        result = result.cpu()

    return {'result': result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict['input'])
    d_model = input_dict.get('d_model', 512)
    nhead = input_dict.get('nhead', 8)
    num_encoder_layers = input_dict.get('num_encoder_layers', 6)
    num_decoder_layers = input_dict.get('num_decoder_layers', 6)
    dim_feedforward = input_dict.get('dim_feedforward', 2048)
    dropout = input_dict.get('dropout', 0.1)
    activation = input_dict.get('activation', 'relu')
    custom_encoder = input_dict.get('custom_encoder', None)
    custom_decoder = input_dict.get('custom_decoder', None)
    layer_norm_eps = input_dict.get('layer_norm_eps', 1e-5)
    batch_first = input_dict.get('batch_first', False)
    norm_first = input_dict.get('norm_first', False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        class TFTransformer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                         dropout, activation, custom_encoder, custom_decoder, layer_norm_eps, batch_first, norm_first):
                super(TFTransformer, self).__init__()
                self.encoder_layers = [TFTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(num_encoder_layers)]
                self.decoder_layers = [TFTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(num_decoder_layers)]
                self.d_model = d_model
                self.nhead = nhead
                self.embedding = tf.keras.layers.Dense(d_model)  # dummy embedding
                self.batch_first = batch_first
                self.layernorm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout_layer = tf.keras.layers.Dropout(dropout)
                self.scale = tf.sqrt(tf.cast(d_model, tf.float32))


            def call(self, src, tgt):
                
                if self.batch_first:
                    src = tf.transpose(src, perm=[1, 0, 2])
                    tgt = tf.transpose(tgt, perm=[1, 0, 2])
                src = self.embedding(src) * self.scale
                tgt = self.embedding(tgt) * self.scale

                for layer in self.encoder_layers:
                    src = layer(src)
                for layer in self.decoder_layers:
                    tgt = layer(tgt, src)  # need encoder output for decoder
                if self.batch_first:
                    tgt = tf.transpose(tgt, perm=[1, 0, 2])

                return self.layernorm(self.dropout_layer(tgt))

        class TFTransformerEncoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(TFTransformerEncoderLayer, self).__init__()
                self.mha = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, value_dim=d_model, attention_axes = [0, 1])
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(dim_feedforward, activation=activation),
                    tf.keras.layers.Dense(d_model)
                ])
                self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout_layer = tf.keras.layers.Dropout(dropout)
                self.norm_first = norm_first

            def call(self, src):
                if self.norm_first:
                    src2 = self.norm1(src)
                    src2 = self.mha(src2, src2, src2)
                    src = src + self.dropout_layer(src2)
                    src2 = self.norm2(src)
                    src2 = self.ffn(src2)
                    src = src + self.dropout_layer(src2)
                else:
                    src2 = self.mha(src, src, src)
                    src = src + self.dropout_layer(src2)
                    src = self.norm1(src)
                    src2 = self.ffn(src)
                    src = src + self.dropout_layer(src2)
                    src = self.norm2(src)
                return src

        class TFTransformerDecoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(TFTransformerDecoderLayer, self).__init__()
                self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, value_dim=d_model, attention_axes = [0, 1])
                self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, value_dim=d_model, attention_axes = [0, 1])
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(dim_feedforward, activation=activation),
                    tf.keras.layers.Dense(d_model)
                ])
                self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.norm3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout_layer = tf.keras.layers.Dropout(dropout)
                self.norm_first = norm_first

            def call(self, tgt, memory):
                if self.norm_first:
                    tgt2 = self.norm1(tgt)
                    tgt2 = self.mha1(tgt2, tgt2, tgt2)
                    tgt = tgt + self.dropout_layer(tgt2)

                    tgt2 = self.norm2(tgt)
                    tgt2 = self.mha2(tgt2, memory, memory)  # Memory is encoder output
                    tgt = tgt + self.dropout_layer(tgt2)

                    tgt2 = self.norm3(tgt)
                    tgt2 = self.ffn(tgt2)
                    tgt = tgt + self.dropout_layer(tgt2)

                else:
                    tgt2 = self.mha1(tgt, tgt, tgt)
                    tgt = tgt + self.dropout_layer(tgt2)
                    tgt = self.norm1(tgt)

                    tgt2 = self.mha2(tgt, memory, memory)
                    tgt = tgt + self.dropout_layer(tgt2)
                    tgt = self.norm2(tgt)

                    tgt2 = self.ffn(tgt)
                    tgt = tgt + self.dropout_layer(tgt2)
                    tgt = self.norm3(tgt)

                return tgt

        transformer_model = TFTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )

        src = input_tensor
        tgt = input_tensor

        result = transformer_model(src, tgt)

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.1
    input_data = {
        'input': np.random.rand(1, 10, 512).astype(np.float32),
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-5,
        'batch_first': True,
        'norm_first': False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()