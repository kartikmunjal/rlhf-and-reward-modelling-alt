import torch

from src.safety.losses import multilabel_loss, prevalence_positive_weights


def test_prevalence_weights_and_cap():
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    raw = prevalence_positive_weights(labels)
    capped = prevalence_positive_weights(labels, cap=2.0)
    assert raw.tolist() == [3.0, 3.0]
    assert capped.tolist() == [2.0, 2.0]


def test_all_preregistered_loss_kinds_are_finite_and_differentiable():
    logits = torch.tensor([[0.2, -0.1], [1.0, -1.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.tensor([2.0, 3.0])
    specifications = [
        {"kind": "bce", "positive_weighting": "none"},
        {"kind": "bce", "positive_weighting": "prevalence"},
        {"kind": "focal", "gamma": 2.0, "alpha": 0.75},
    ]
    for specification in specifications:
        loss = multilabel_loss(logits, labels, specification, weights)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
