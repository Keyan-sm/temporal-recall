"""Use the bitemporal memory store directly: facts that evolve, recalled as-of any time.

Run:  pip install temporal-recall  &&  python examples/quickstart.py
"""
from temporal_recall import Memory

mem = Memory(policy="bitemporal")
# An agent learns, over time, where someone works. Each fact is stored with the time
# it became true — not overwritten.
mem.remember("Lina works at Acme.", valid_time=2019, object_value="Acme")
mem.remember("Lina works at Globex.", valid_time=2022, object_value="Globex")
mem.remember("Lina works at Initech.", valid_time=2025, object_value="Initech")
mem.index()

q = "Where does Lina work?"
print("as-of 2020 :", mem.recall(q, as_of=2020)[0].object_value)   # Acme
print("as-of 2023 :", mem.recall(q, as_of=2023)[0].object_value)   # Globex
print("today      :", mem.recall(q, as_of=2026)[0].object_value)   # Initech

# A 'latest-wins' store (what most agent memories do) would answer 'Initech' for all
# three — it cannot tell you what was true in the past. That gap is what this library
# measures (see `temporal-recall benchmark`) and fixes.
