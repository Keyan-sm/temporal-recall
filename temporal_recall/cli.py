"""Command line: ``temporal-recall benchmark | demo``."""
from __future__ import annotations

import argparse


def _cmd_benchmark(args) -> None:
    from .benchmark import run_benchmark
    from .data import load_templama

    data = load_templama(args.split)
    report = run_benchmark(data, k=args.k, n_per_kind=args.n, seed=args.seed)
    _print_report(report)


def _cmd_demo(args) -> None:
    """A tiny, no-download illustration of the three policies on one evolving fact."""
    from .memory import Memory

    facts = [
        ("Cristiano Ronaldo plays for Manchester United.", 2008, "Manchester United"),
        ("Cristiano Ronaldo plays for Real Madrid.", 2010, "Real Madrid"),
        ("Cristiano Ronaldo plays for Juventus.", 2018, "Juventus"),
        ("Cristiano Ronaldo plays for Al Nassr.", 2023, "Al Nassr"),
    ]
    query = "Cristiano Ronaldo plays for _X_."
    for policy in ("flat", "latest", "bitemporal"):
        mem = Memory(policy=policy)
        for text, year, obj in facts:
            mem.remember(text, valid_time=year, object_value=obj)
        mem.index()
        as_of_2012 = mem.recall(query, as_of=2012)
        current = mem.recall(query, as_of=2024)
        print(f"policy={policy:<11}  as-of 2012 -> {_obj(as_of_2012):<18}  current -> {_obj(current)}")
    print("\n(ground truth: as-of 2012 = Real Madrid; current = Al Nassr)")


def _obj(records) -> str:
    return records[0].object_value if records else "(none)"


def _print_report(report) -> None:
    print(f"TempLAMA temporal-recall benchmark  |  statements={report.n_statements:,}  k={report.k}")
    print(f"queries: as_of={report.n_queries['as_of']:,}  current={report.n_queries['current']:,}")
    print(f"retrieval ceiling (gold answer was retrievable): "
          f"as-of={report.retrieval_ceiling['as_of']:.3f}  current={report.retrieval_ceiling['current']:.3f}")
    print("\n  policy        as-of      current")
    for policy in ("flat", "latest", "bitemporal"):
        acc = report.accuracy[policy]
        print(f"  {policy:<11}  {acc['as_of']:.3f}      {acc['current']:.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="temporal-recall",
                                     description="Benchmark the temporal correctness of agent memory.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("benchmark", help="Run the TempLAMA temporal-recall benchmark.")
    p_bench.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_bench.add_argument("-k", type=int, default=20, help="retrieval depth")
    p_bench.add_argument("-n", type=int, default=3000, help="queries per kind")
    p_bench.add_argument("--seed", type=int, default=0)

    sub.add_parser("demo", help="Illustrate the three policies on one evolving fact (no download).")

    args = parser.parse_args(argv)
    {"benchmark": _cmd_benchmark, "demo": _cmd_demo}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
