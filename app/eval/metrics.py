"""Ranking metrics: Precision@K, Recall@K, NDCG@K.

`ranked` is an ordered list of item ids (best first). `relevant` is the set of
ground-truth relevant ids. NDCG uses binary relevance (gain 1 if relevant).
"""
import math


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(ranked[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 2)  # rank position idx+1 -> log2(idx+2)
    return dcg


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def evaluate(ranked: list, relevant: set, ks: list[int]) -> dict:
    out = {}
    for k in ks:
        out[f"precision@{k}"] = round(precision_at_k(ranked, relevant, k), 4)
        out[f"recall@{k}"] = round(recall_at_k(ranked, relevant, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(ranked, relevant, k), 4)
    return out
