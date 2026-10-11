import argparse
from pathlib import Path

from . import data, experiment

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 2026
PER_CATEGORY = 300
PILOT_PER_CATEGORY = 8


def sample() -> list[data.Item]:
    data.download(DATA)
    items = data.load(DATA)
    manifest = DATA / "sample.json"
    if not manifest.exists():
        data.write_manifest(manifest, data.stratified(items, PER_CATEGORY, SEED), SEED, PER_CATEGORY)
    return data.from_manifest(manifest, items)


def pilot() -> list[data.Item]:
    used = {it.id for it in sample()}
    items = data.load(DATA)
    return data.stratified([it for it in items if it.id not in used], PILOT_PER_CATEGORY, SEED)


def cmd_run(args):
    configs = [c for c in experiment.CONFIGS if c.family in args.families]
    items = sample() if args.split == "test" else pilot()
    results = RESULTS if args.split == "test" else RESULTS / "pilot"
    if args.formats:
        configs = [
            experiment.Config(c.family, c.label, c.prefix, tuple(f for f in c.formats if f in args.formats))
            for c in configs
        ]
    experiment.run(configs, items, pilot(), results, set(args.keep))


def cmd_rebench(args):
    configs = [
        experiment.Config(c.family, c.label, c.prefix, tuple(f for f in c.formats if f in args.formats))
        for c in experiment.CONFIGS
        if c.family in args.families
    ]
    experiment.rebench(configs, pilot(), RESULTS, set(args.keep))


def cmd_analyze(_):
    from . import analysis

    analysis.main(sample(), RESULTS, ROOT / "figures")


def main():
    parser = argparse.ArgumentParser(prog="uaquant")
    sub = parser.add_subparsers(required=True)
    run = sub.add_parser("run")
    run.add_argument("split", choices=["pilot", "test"])
    run.add_argument("--families", nargs="+", default=[c.family for c in experiment.CONFIGS])
    run.add_argument("--formats", nargs="+")
    run.add_argument("--keep", nargs="*", default=[], help="Ollama tags not to delete after the run")
    run.set_defaults(func=cmd_run)
    rebench = sub.add_parser("rebench")
    rebench.add_argument("--families", nargs="+", required=True)
    rebench.add_argument("--formats", nargs="+", required=True)
    rebench.add_argument("--keep", nargs="*", default=[])
    rebench.set_defaults(func=cmd_rebench)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
