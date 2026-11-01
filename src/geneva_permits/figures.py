"""Report figures: a PDF and a PNG per figure, without timestamps so reruns match byte for byte."""

from geneva_permits.config import FIGURES_DIR

SUFFIXES = ("pdf", "png")


def save_figure(fig, name, directory=None):
    """Write <name>.pdf and <name>.png into the figures directory; the paths written."""
    directory = directory or FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in SUFFIXES:
        metadata = {"CreationDate": None} if suffix == "pdf" else None  # no timestamp in the PDF
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight", metadata=metadata)
    return [directory / f"{name}.{suffix}" for suffix in SUFFIXES]
