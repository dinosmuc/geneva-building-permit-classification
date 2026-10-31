"""CamemBERT fine-tuning on a toy model and tokenizer, into a throwaway store."""

import mlflow
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    PreTrainedTokenizerFast,
)

from geneva_permits import camembert
from geneva_permits.evaluate import CLASSES

WORDS = ["villa", "bureaux", "immeuble", "route", "jardin", "toiture", "garage", "usine", "hangar"]
TRAIN_LABELS = [
    "VIL",
    "REN",
    "NOU",
    "EQU",
    "PI",
    "AGR",
    "AGV",
    "AFF",
    "REN",
]  # every class, REN twice
TEXTS = pd.DataFrame(
    {
        "partition": ["train_original"] * 9 + ["validation"] * 4,
        "label": TRAIN_LABELS + ["VIL", "REN", "NOU", "AFF"],
        "text": [f"{w} {w}" for w in WORDS] + ["villa", "bureaux", "immeuble", "usine"],
    }
)
SETTINGS = {
    "seeds": [13],
    "precision": "fp16",
    "max_seq_length": 4,
    "batch_size": 4,
    "epochs": 2,
    "pilot_learning_rates": [1e-3, 1e-4],
    "learning_rate": 1e-3,
    "schedule": "linear",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "adam_betas": [0.9, 0.999],
    "adam_epsilon": 1e-6,
    "max_grad_norm": 1.0,
}


def toy_tokenizer():
    vocab = {token: i for i, token in enumerate(["[PAD]", "[CLS]", "[SEP]", "[UNK]", *WORDS])}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, **camembert.SPECIAL_TOKENS)


def toy_model(_settings):
    config = DebertaV2Config(
        vocab_size=4 + len(WORDS),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=8,
        num_labels=len(CLASSES),
        pad_token_id=0,
    )
    return DebertaV2ForSequenceClassification(config)


@pytest.fixture
def toy(monkeypatch, mlruns):
    monkeypatch.setattr(camembert, "load_tokenizer", lambda _settings: toy_tokenizer())
    monkeypatch.setattr(camembert, "load_model", toy_model)


def test_encode_cuts_at_max_seq_length_and_maps_labels_to_class_positions():
    features = camembert.encode(toy_tokenizer(), TEXTS.head(2), max_seq_length=3)
    assert [f["input_ids"] for f in features] == [[1, 4, 2], [1, 5, 2]]
    assert [f["labels"] for f in features] == [CLASSES.index("VIL"), CLASSES.index("REN")]


def test_loss_weights_follow_the_class_order():
    labels = pd.Series(TRAIN_LABELS)
    weights = camembert.loss_weights(labels, "full", torch.device("cpu"))
    assert weights.shape == (len(CLASSES),)
    assert weights[CLASSES.index("REN")] == pytest.approx(9 / (8 * 2))  # n / (K n_c)
    assert weights[CLASSES.index("AFF")] == pytest.approx(9 / (8 * 1))
    assert torch.all(camembert.loss_weights(labels, "none", torch.device("cpu")) == 1)
    with pytest.raises(KeyError):
        camembert.loss_weights(labels.head(3), "none", torch.device("cpu"))


def test_main_runs_every_weighting_and_seed_and_stores_the_model(toy, mlruns):
    results = camembert.run_main(TEXTS, SETTINGS)
    assert list(results["weighting"]) == ["none", "medium", "full"]
    assert results["macro_f1"].between(0, 1).all()
    run = mlflow.search_runs(experiment_names=["selection"]).iloc[0]
    assert run["params.model"] == "camembert"
    assert run["params.stage"] == "main"
    assert run["params.precision"] == ("fp16" if torch.cuda.is_available() else "fp32")
    client = mlflow.MlflowClient()
    assert len(client.get_metric_history(run["run_id"], "val_macro_f1")) == SETTINGS["epochs"]
    assert any(mlruns.rglob("model/model.safetensors"))
    assert camembert.selected() in camembert.OPTIONS


def test_pilot_runs_one_learning_rate_at_a_time_without_weighting(toy):
    results = camembert.run_pilot(TEXTS, SETTINGS)
    assert list(results["learning_rate"]) == SETTINGS["pilot_learning_rates"]
    assert set(results["weighting"]) == {"none"}
    runs = mlflow.search_runs(experiment_names=["selection"])
    assert set(runs["params.stage"]) == {"pilot"}
    assert set(runs["params.seed"]) == {"13"}


def test_main_refuses_an_unfrozen_learning_rate():
    with pytest.raises(ValueError, match="learning_rate is null"):
        camembert.run_main(TEXTS, SETTINGS | {"learning_rate": None})


def test_best_weighting_averages_over_seeds_and_keeps_the_earlier_option_on_ties():
    main_runs = pd.DataFrame(
        {
            "params.weighting": ["full", "full", "medium", "medium", "none", "none"],
            "metrics.macro_f1": [0.5, 0.7, 0.6, 0.6, 0.7, 0.5],
        }
    )
    assert camembert.best_weighting(main_runs) == "none"  # 0.6 everywhere: none < medium < full
    main_runs.loc[0, "metrics.macro_f1"] = 0.9
    assert camembert.best_weighting(main_runs) == "full"
