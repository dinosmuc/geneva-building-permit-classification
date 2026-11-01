"""Figures are written as PDF and PNG, and the PDF is the same byte for byte on every save."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geneva_permits.figures import save_figure


def test_save_figure_writes_both_formats_reproducibly(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    first = save_figure(fig, "probe", tmp_path / "a")
    second = save_figure(fig, "probe", tmp_path / "b")
    assert [p.name for p in first] == ["probe.pdf", "probe.png"]
    assert all(p.is_file() for p in first + second)
    assert first[0].read_bytes() == second[0].read_bytes()
    plt.close(fig)
