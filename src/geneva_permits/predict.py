"""Categories for new descriptions from the stored CamemBERT model (config.yaml `prediction`)."""

import pandas as pd
import torch

from geneva_permits import camembert
from geneva_permits.config import load_config
from geneva_permits.evaluate import CLASSES


def load(config):
    """The configured stored model and its tokenizer, on the available device."""
    choice = config["prediction"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = camembert.stored_run(choice["weighting"], choice["seed"])
    model = camembert.load_stored(run_id).to(device)
    return model, camembert.load_tokenizer(config["models"]["primary"]), device


def classify(descriptions, config=None):
    """One row per description: the predicted class and the softmax score of every class."""
    config = config or load_config()
    model, tokenizer, device = load(config)
    texts = pd.DataFrame({"text": list(descriptions)})
    loader = camembert.batches(tokenizer, texts, config["training"])
    scores = camembert.logits(model, loader, device, config["training"]["precision"]).softmax(-1)
    table = pd.DataFrame(scores.numpy(), columns=CLASSES, index=texts["text"])
    table.insert(0, "predicted", table.idxmax(axis=1))
    return table
