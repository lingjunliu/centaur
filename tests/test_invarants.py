import pytest
from learner.invariant_inference import *
from learner.inputs import get_inputs

@pytest.mark.unit
def test_scatter_invariants():
    api = "scatter"
    ruleset = infer_invariants(api, get_inputs(api))
    ref_ruleset = set([
                        ('rule_2', 'input', 'dim'),
                        ('rule_2', 'index', 'dim'),
                        ('rule_2', 'src', 'dim'),
                        ('rule_3', 'input', 'src'),
                        ('rule_3', 'input', 'index'),
                        ('rule_3', 'src', 'index'),
                        ('rule_4', 'input', 'src')
                    ])
    
    for rule_tuple in ref_ruleset:
        assert len(ruleset.intersection(set([rule_tuple]))) > 0, f"{rule_tuple} expected to be inferred, but it was not"

@pytest.mark.unit
def test_conv_transpose2d_invariants():
    api = "conv_transpose2d"
    ruleset = infer_invariants(api, get_inputs(api))
    ref_ruleset = set([
                        ('rule_3', 'input', 'weight'),
                        ('rule_4', 'input', 'weight'),
                        ('rule_7', 'weight', 'input')
                    ])
    for rule_tuple in ref_ruleset:
        assert len(ruleset.intersection(set([rule_tuple]))) > 0, f"{rule_tuple} expected to be inferred, but it was not"

if __name__ == "__main__":
    test_scatter_invariants()