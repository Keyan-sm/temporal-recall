from temporal_recall import Memory, Record, select


def _candidates():
    # Same fact asserted at three times (most-similar-first ordering is irrelevant here;
    # the policies sort by valid_time themselves).
    return [
        Record(text="X plays for A.", valid_time=2008, object_value="A", aliases={"A"}),
        Record(text="X plays for B.", valid_time=2012, object_value="B", aliases={"B"}),
        Record(text="X plays for C.", valid_time=2020, object_value="C", aliases={"C"}),
    ]


def test_flat_ignores_time_returns_top_candidate():
    cands = _candidates()
    assert select("flat", cands, as_of=2010)[0].object_value == "A"  # first = top similarity


def test_latest_returns_most_recent_regardless_of_as_of():
    cands = _candidates()
    assert select("latest", cands, as_of=2010)[0].object_value == "C"


def test_bitemporal_returns_version_in_effect_at_as_of():
    cands = _candidates()
    assert select("bitemporal", cands, as_of=2010)[0].object_value == "A"  # latest <= 2010
    assert select("bitemporal", cands, as_of=2015)[0].object_value == "B"
    assert select("bitemporal", cands, as_of=2025)[0].object_value == "C"


def test_bitemporal_before_first_assertion_falls_back_to_earliest():
    cands = _candidates()
    assert select("bitemporal", cands, as_of=2000)[0].object_value == "A"


def test_memory_store_recall_end_to_end():
    mem = Memory(policy="bitemporal")
    for text, year, obj in [("Ronaldo plays for Sporting.", 2003, "Sporting"),
                            ("Ronaldo plays for United.", 2006, "United"),
                            ("Ronaldo plays for Madrid.", 2012, "Madrid")]:
        mem.remember(text, valid_time=year, object_value=obj)
    mem.index()
    assert mem.recall("Ronaldo plays for _X_.", as_of=2008)[0].object_value == "United"
    assert mem.recall("Ronaldo plays for _X_.", as_of=2020)[0].object_value == "Madrid"
