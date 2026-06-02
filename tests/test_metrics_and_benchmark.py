from temporal_recall import run_benchmark
from temporal_recall import metrics
from temporal_recall.data import Statement, TempLama


def test_normalize_and_alias_match():
    assert metrics.normalize("The New England Patriots!") == "new england patriots"
    assert metrics.matches(["Patriots"], ["The Patriots", "New England Patriots"])
    assert not metrics.matches(["Real Madrid"], ["Juventus", "Al Nassr"])


def _toy_templama() -> TempLama:
    """Three players, each changing clubs across 2010-2012 (an evolving fact each)."""
    plan = {
        "P1": [("Adams", 2010, "Reds"), ("Adams", 2011, "Blues"), ("Adams", 2012, "Greens")],
        "P2": [("Baker", 2010, "Lions"), ("Baker", 2011, "Lions"), ("Baker", 2012, "Tigers")],
        "P3": [("Cole", 2010, "Hawks"), ("Cole", 2011, "Eagles"), ("Cole", 2012, "Eagles")],
    }
    statements, timeline = [], {}
    for key, rows in plan.items():
        timeline[key] = {}
        for subj, year, obj in rows:
            query = f"{subj} plays for _X_."
            statements.append(Statement(fact_key=key, query=query, year=year,
                                        object_name=obj, aliases={obj}))
            timeline[key][year] = {obj}
    return TempLama(statements=statements, timeline=timeline)


def test_benchmark_bitemporal_beats_naive_on_as_of():
    report = run_benchmark(_toy_templama(), k=10, n_per_kind=50, seed=0)
    acc = report.accuracy
    # Retrieval should surface the right fact; ceilings near 1.0.
    assert report.retrieval_ceiling["as_of"] > 0.9
    # Bitemporal handles as-of; latest-wins (replacement) does materially worse on as-of.
    assert acc["bitemporal"]["as_of"] > 0.95
    assert acc["bitemporal"]["as_of"] > acc["latest"]["as_of"] + 0.1
    # Latest-wins is strong on 'current' (it returns the latest value).
    assert acc["latest"]["current"] > 0.95
    for policy in ("flat", "latest", "bitemporal"):
        for kind in ("as_of", "current"):
            assert 0.0 <= acc[policy][kind] <= 1.0
