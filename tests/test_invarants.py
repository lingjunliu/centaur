import pytest
from learner.invariant_inference import *
from learner.inputs import get_inputs

@pytest.mark.unit
def test_scatter_invariants():
    api = "scatter"
    ruleset = infer_invariants(api)[0]
    ref_ruleset = set([
                        (2, 'rule_2', 'input', 'dim'),
                        (2, 'rule_2', 'index', 'dim'),
                        (2, 'rule_2', 'src', 'dim'),
                        (2, 'rule_3', 'input', 'src'),
                        (2, 'rule_3', 'input', 'index'),
                        (2, 'rule_3', 'src', 'index'),
                        (2, 'rule_4', 'input', 'src')
                    ])
    
    for rule_tuple in ref_ruleset:
        assert len(ruleset.intersection(set([rule_tuple]))) > 0, f"{rule_tuple} expected to be inferred, but it was not"

@pytest.mark.unit
def test_conv_transpose2d_invariants():
    api = "conv_transpose2d"
    ruleset = infer_invariants(api)[0]
    ref_ruleset = set([
                        (2, 'rule_3', 'input', 'weight'),
                        (2, 'rule_4', 'input', 'weight'),
                        (2, 'rule_7', 'weight', 'input')
                    ])
    for rule_tuple in ref_ruleset:
        assert len(ruleset.intersection(set([rule_tuple]))) > 0, f"{rule_tuple} expected to be inferred, but it was not"

if __name__ == "__main__":
    test_scatter_invariants()