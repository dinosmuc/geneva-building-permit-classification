"""The `permits` command line dispatches to the snapshot functions and reports failures."""

from pathlib import Path

import pytest

from geneva_permits import cli, source


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["data", "fetch"], cli.data_fetch),
        (["data", "verify"], cli.data_verify),
        (["data", "fetch-live"], cli.data_fetch_live),
    ],
)
def test_commands_route_to_their_handlers(argv, handler):
    assert cli.build_parser().parse_args(argv).handler is handler


def test_an_action_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["data"])


def test_snapshot_errors_exit_with_status_one(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise source.SnapshotError("archive does not match the lock")

    monkeypatch.setattr(source, "verify_snapshot", fail)
    assert cli.main(["data", "verify"]) == 1
    assert "does not match the lock" in capsys.readouterr().err


def test_fetch_reports_the_action(monkeypatch, capsys):
    monkeypatch.setattr(source, "fetch_snapshot", lambda: (Path("pinned.zip"), "kept"))
    assert cli.main(["data", "fetch"]) == 0
    assert capsys.readouterr().out.strip() == "kept: pinned.zip"
