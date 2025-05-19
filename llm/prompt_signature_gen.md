For the api {api}, provide a signature of the api as a dictionary. Here are the specifications:
- The keys should be the name of the argument (e.g. 'input')
- The values should be the type of argument it accepts. This value can only be (strictly) one of the following options:
    - "tensor"
    - "tensor_list"
    - "integer"
    - "float"
    - "boolean"
    - "string"
    - "dtype"
    - "list"
    - "tuple"
- Choose the value that most closely matches with the argument type. For example, change IntTensor or FloatTensor type arguments to "tensor", change "Numbers" to "float" etc.
- If you are confused about a type, choose the best one and add a small comment on what you think should have been there instead
- Do not add the arguments "out" or "device"
- If the api has multiple signatures, choose the one used most commonly. Same goes for argument types, if an argument can be of multiple types (e.g. tuples and integers), choose the most commonly used one
- Assume I already have a dictionary called `signatures` (you don't have to initialize it). Provide the output as assigning the signature to the entry of the `signatures` dictionary where the key is the api name. Examples:
```python
signatures["torch.scatter_add"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor"
}
signatures["multilabel_soft_margin_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
```
- Only provide the code, skip any other text. Do not include verbose comments inside code other than short comments about confusion. Do not make any system calls within the code. Do not break any dependencies.