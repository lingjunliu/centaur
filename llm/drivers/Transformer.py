import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["src"], requires_grad=False)
    num_layers = input_dict.get("num_layers", 6)
    d_model = input_dict.get("d_model", 512)
    nhead = input_dict.get("nhead", 8)
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    custom_encoder = input_dict.get("custom_encoder", None)
    custom_decoder = input_dict.get("custom_decoder", None)
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    transformer_layer = torch.nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
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
    tgt = torch.rand_like(src, requires_grad=False)

    if not cpu:
        tgt = tgt.cuda()
    
    result = transformer_layer(src, tgt)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().cpu().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["src"], dtype=tf.float32)
        num_layers = input_dict.get("num_layers", 6)
        d_model = input_dict.get("d_model", 512)
        nhead = input_dict.get("nhead", 8)
        dim_feedforward = input_dict.get("dim_feedforward", 2048)
        dropout = input_dict.get("dropout", 0.1)
        activation = input_dict.get("activation", "relu")
        custom_encoder = input_dict.get("custom_encoder", None)
        custom_decoder = input_dict.get("custom_decoder", None)
        layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
        batch_first = input_dict.get("batch_first", False)
        norm_first = input_dict.get("norm_first", False)
    
        class FeedForward(tf.keras.layers.Layer):
            def __init__(self, d_model, dim_feedforward, dropout, activation):
                super(FeedForward, self).__init__()
                self.dense1 = tf.keras.layers.Dense(dim_feedforward)
                self.dense2 = tf.keras.layers.Dense(d_model)
                self.dropout = tf.keras.layers.Dropout(dropout)
                self.activation = tf.keras.activations.get(activation) if isinstance(activation, str) else activation

            def call(self, x):
                x = self.dense1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.dense2(x)
                return x
        
        class TFTransformerLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(TFTransformerLayer, self).__init__()
                self.mha = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
                self.ffn = FeedForward(d_model, dim_feedforward, dropout, activation)
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout1 = tf.keras.layers.Dropout(dropout)
                self.dropout2 = tf.keras.layers.Dropout(dropout)
                self.norm_first = norm_first

            def call(self, x, y): # x = src, y = tgt
                attn_output = self.mha(x, y, y)
                if self.norm_first:
                    x_norm = self.layernorm1(x)
                    x = x + self.dropout1(attn_output)
                    x_norm = self.layernorm2(x)
                    ffn_output = self.ffn(x_norm)
                    x = x + self.dropout2(ffn_output)
                else:
                    x = self.layernorm1(x + self.dropout1(attn_output))
                    ffn_output = self.ffn(x)
                    x = self.layernorm2(x + self.dropout2(ffn_output))
                return x

        class TFTransformer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first):
                super(TFTransformer, self).__init__()
                self.encoder_layers = [TFTransformerLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(num_encoder_layers)]
                self.decoder_layers = [TFTransformerLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(num_decoder_layers)]
                self.batch_first = batch_first
                self.d_model = d_model

            def call(self, src, tgt):
                encoder_output = src
                decoder_output = tgt
                for enc_layer in self.encoder_layers:
                    encoder_output = enc_layer(encoder_output, encoder_output) # using src as both x and y
                
                for dec_layer in self.decoder_layers:
                    decoder_output = dec_layer(decoder_output, encoder_output) # passing decoder_output and encoder output as x and y

                return decoder_output

        transformer_layer = TFTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )

        src = input_tensor
        tgt = tf.random.uniform(shape=tf.shape(src), dtype=tf.float32)

        result = transformer_layer(src, tgt)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "src": np.random.rand(2, 3, 512).astype(np.float32),
        "num_layers": 2,
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "batch_first": False,
        "norm_first": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()