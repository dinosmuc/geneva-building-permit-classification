"""Command line: `permits <area> <action>`."""

import argparse
import sys

from geneva_permits import baseline, camembert, classical, curves, preprocess, source, splits


def data_fetch(args):
    path, action = source.fetch_snapshot()
    print(f"{action}: {path}")


def data_verify(args):
    path = source.verify_snapshot()
    print(f"verified: {path} (SHA-256 {source.read_lock()['sha256']})")


def data_splits(args):
    path, rows = splits.write_splits()
    print(f"wrote {rows:,} retained rows: {path}")


def data_preprocess(args):
    path, rows = preprocess.write_texts()
    print(f"wrote {rows:,} lemmatized rows: {path}")


def train_majority(args):
    scores = baseline.run_majority()
    macro_f1, accuracy = scores["macro_f1"], scores["accuracy"]
    print(f"majority on validation: macro-F1 {macro_f1:.3f}, accuracy {accuracy:.1%}")


def train_linearsvc(args):
    best = classical.select(classical.run_grid(args.variant))
    print(
        f"linearsvc {args.variant} on validation: best macro-F1 {best['macro_f1']:.3f} at "
        f"weighting={best['weighting']} C=2^{best['C_log2']} ngrams={best['ngram_range']}"
    )


def train_curves(args):
    variant, weighting = classical.selected()
    best = curves.run_curves(variant, weighting)
    print(f"learning curves of linearsvc {variant}, weighting {weighting}: best macro-F1 per draw")
    print(best[["fraction", "seed", "train_rows", "C_log2", "ngram_range", "macro_f1"]].to_string())


def train_camembert(args):
    if args.stage == camembert.PILOT:
        results = camembert.run_pilot()
        best = results.loc[results["macro_f1"].idxmax()]
        print(f"camembert pilot: best macro-F1 {best['macro_f1']:.3f} at {best['learning_rate']}")
    else:
        results = camembert.run_main()
        print(f"camembert main: selected weighting {camembert.selected()}")
    print(results.to_string())


def build_parser():
    parser = argparse.ArgumentParser(prog="permits", description=__doc__)
    areas = parser.add_subparsers(dest="area", required=True)

    data = areas.add_parser("data", help="the pinned SITG source snapshot")
    actions = data.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "fetch", help="make the pinned archive available, downloading it from the mirror"
    ).set_defaults(handler=data_fetch)
    actions.add_parser("verify", help="check the pinned archive against the lock").set_defaults(
        handler=data_verify
    )
    actions.add_parser(
        "splits", help="store the retained split in data/splits/splits.csv"
    ).set_defaults(handler=data_splits)
    actions.add_parser(
        "preprocess", help="store the lemmatized texts in data/processed/texts.parquet"
    ).set_defaults(handler=data_preprocess)

    train = areas.add_parser("train", help="fit on the training rows, score on the validation rows")
    actions = train.add_subparsers(dest="action", required=True)
    actions.add_parser("majority", help="the majority-class baseline").set_defaults(
        handler=train_majority
    )
    linearsvc = actions.add_parser("linearsvc", help="TF-IDF + LinearSVC over the whole grid")
    linearsvc.add_argument("variant", choices=classical.VARIANTS, help="A: text, B: lemmas")
    linearsvc.set_defaults(handler=train_linearsvc)
    actions.add_parser(
        "curves", help="learning curves of the selected LinearSVC variant and weighting"
    ).set_defaults(handler=train_curves)
    transformer = actions.add_parser("camembert", help="CamemBERTav2 fine-tuning")
    transformer.add_argument(
        "stage",
        choices=[camembert.PILOT, camembert.MAIN],
        help="pilot: learning rates; main: 9 runs",
    )
    transformer.set_defaults(handler=train_camembert)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except source.SnapshotError as error:
        print(f"permits: {error}", file=sys.stderr)
        return 1
    return 0
