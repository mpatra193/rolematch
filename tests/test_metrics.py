from app.eval.metrics import evaluate, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_and_recall():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "x"}
    assert precision_at_k(ranked, relevant, 2) == 0.5  # a hit, b miss
    assert precision_at_k(ranked, relevant, 4) == 0.5  # a, c hit of 4
    assert recall_at_k(ranked, relevant, 4) == round(2 / 3, 10) or abs(recall_at_k(ranked, relevant, 4) - 2 / 3) < 1e-9


def test_ndcg_perfect_is_one():
    ranked = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(ranked, relevant, 3) == 1.0


def test_ndcg_rewards_order():
    relevant = {"a"}
    top = ndcg_at_k(["a", "b", "c"], relevant, 3)
    low = ndcg_at_k(["b", "c", "a"], relevant, 3)
    assert top > low


def test_evaluate_keys():
    out = evaluate(["a"], {"a"}, [1, 3])
    assert set(out) == {"precision@1", "recall@1", "ndcg@1", "precision@3", "recall@3", "ndcg@3"}
