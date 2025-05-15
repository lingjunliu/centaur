import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.attention import SDPParams

    input_tensor = torch.tensor(input_dict["embed_dim"])
    dropout = input_dict.get("dropout", 0.0)
    attention_dropout = input_dict.get("attention_dropout", 0.0)
    sequence_dropout = input_dict.get("sequence_dropout", None)
    is_causal = input_dict.get("is_causal", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    params = SDPParams(embed_dim=int(input_tensor.item()), dropout=dropout, attention_dropout=attention_dropout,
                       sequence_dropout=sequence_dropout, is_causal=is_causal)

    result = params.embed_dim, params.dropout, params.attention_dropout, params.sequence_dropout, params.is_causal

    if not cpu:
        result = (r.cpu() if isinstance(r, torch.Tensor) else r for r in result)

    return {"embed_dim": np.array(result[0].item()),
            "dropout": result[1],
            "attention_dropout": result[2],
            "sequence_dropout": result[3],
            "is_causal": result[4]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    embed_dim = input_dict["embed_dim"]
    dropout = input_dict.get("dropout", 0.0)
    attention_dropout = input_dict.get("attention_dropout", 0.0)
    sequence_dropout = input_dict.get("sequence_dropout", None)
    is_causal = input_dict.get("is_causal", False)

    return {"embed_dim": embed_dim.item(),
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "sequence_dropout": sequence_dropout,
            "is_causal": is_causal}

def main():
    A_TOL = 0.01

    input_data = {
        "embed_dim": np.array(128, dtype=np.int64),
        "dropout": 0.1,
        "attention_dropout": 0.05,
        "sequence_dropout": None,
        "is_causal": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["embed_dim"], tf_result["embed_dim"], atol=A_TOL), "embed_dim do not match"
    assert np.isclose(torch_result["dropout"], tf_result["dropout"], atol=A_TOL), "dropout do not match"
    assert np.isclose(torch_result["attention_dropout"], tf_result["attention_dropout"], atol=A_TOL), "attention_dropout do not match"
    assert torch_result["sequence_dropout"] == tf_result["sequence_dropout"], "sequence_dropout do not match"
    assert torch_result["is_causal"] == tf_result["is_causal"], "is_causal do not match"

    print("Success")

if __name__ == "__main__":
    main()