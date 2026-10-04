"""Command line: `permits <area> <action>`."""

import argparse
import sys

from geneva_permits import source


def data_fetch(args):
    path, action = source.fetch_snapshot()
    print(f"{action}: {path}")


def data_verify(args):
    path = source.verify_snapshot()
    print(f"verified: {path} (SHA-256 {source.read_lock()['sha256']})")


def data_fetch_live(args):
    archive, manifest = source.fetch_live_snapshot()
    state = "identical to" if manifest["matches_pin"] else "differs from"
    print(f"downloaded: {archive} (SHA-256 {manifest['sha256']}, {state} the pin)")


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
        "fetch-live", help="download today's SITG export into data/raw/live/ (never the pin)"
    ).set_defaults(handler=data_fetch_live)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except source.SnapshotError as error:
        print(f"permits: {error}", file=sys.stderr)
        return 1
    return 0
