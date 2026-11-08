"""CamemBERTav2 fine-tuning: class weights in the loss, one MLflow run per weighting and seed.

The pinned revision ships a WordPiece tokenizer.json but names RobertaTokenizer in its
tokenizer_config.json, so AutoTokenizer splits French words into single characters and
DebertaV2TokenizerFast fails to load it (transformers 5.17). The tokenizer is therefore built
from tokenizer.json directly. Words longer than 16 characters become [UNK] (its
max_input_chars_per_word): 178 training tokens, e.g. "télécommunication".
"""

import tempfile
import time

import mlflow
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    get_scheduler,
    set_seed,
)

from geneva_permits.config import load_config
from geneva_permits.curves import draws, nested_subsets
from geneva_permits.evaluate import CLASSES, scores
from geneva_permits.identity import current_identity, fingerprint
from geneva_permits.preprocess import load_texts, train_validation
from geneva_permits.tracking import (
    LEARNING_CURVES,
    SELECTION,
    camembert_runs,
    log_evaluation,
    runs,
    set_experiment,
    start_run,
)
from geneva_permits.weights import OPTIONS, class_weights

# special_tokens_map.json of the pinned revision
SPECIAL_TOKENS = {
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "mask_token": "[MASK]",
}
PILOT = "pilot"
MAIN = "main"
CURVES = "curves"
EXPERIMENT = {PILOT: SELECTION, MAIN: SELECTION, CURVES: LEARNING_CURVES}

POINT = {"weighting": str, "seed": int, "learning_rate": float, "fraction": float}
CHECKPOINTS = ["best", "final"]
LOGGED = [
    "max_seq_length",
    "batch_size",
    "epochs",
    "checkpoint",
    "schedule",
    "warmup_ratio",
    "weight_decay",
]


def load_tokenizer(model):
    """The pinned WordPiece tokenizer, from tokenizer.json (see the module docstring)."""
    path = hf_hub_download(model["id"], "tokenizer.json", revision=model["revision"])
    return PreTrainedTokenizerFast(tokenizer_file=path, **SPECIAL_TOKENS)


def load_model(model):
    """The pinned encoder with a fresh classification head over the eight classes."""
    return AutoModelForSequenceClassification.from_pretrained(
        model["id"], revision=model["revision"], num_labels=len(CLASSES)
    )


def encode(tokenizer, texts, max_seq_length):
    """One feature dict per row: token ids cut at max_seq_length, label id = position in CLASSES."""
    encoded = tokenizer(texts["text"].tolist(), truncation=True, max_length=max_seq_length)
    features = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"], strict=True)
    ]
    if "label" in texts:  # new descriptions to classify carry none
        for feature, label in zip(features, texts["label"].map(CLASSES.index), strict=True):
            feature["labels"] = label
    return features


def batches(tokenizer, texts, settings, shuffle=False):
    """The rows encoded and batched, padded to the longest row of each batch."""
    return DataLoader(
        encode(tokenizer, texts, settings["max_seq_length"]),
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def loss_weights(labels, weighting, device):
    """The class weights of the rows trained on, in CLASSES order, for CrossEntropyLoss."""
    weights = class_weights(labels, weighting)[CLASSES].to_numpy()  # KeyError if a class is missing
    return torch.tensor(weights, dtype=torch.float32, device=device)


def mixed_precision(device, precision):
    """fp16 only on the GPU; the CPU, where the tests run, trains in fp32."""
    return precision == "fp16" and device.type == "cuda"


def autocast(device, precision):
    """fp16 autocast when mixed_precision holds, otherwise a no-op."""
    enabled = mixed_precision(device, precision)
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled)


def logits(model, loader, device, precision):
    """The logits of every row in the loader, stacked on the CPU."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with autocast(device, precision):
                outputs.append(model(**batch).logits.float().cpu())
    return torch.cat(outputs)


def predict(model, loader, device, precision):
    """Predicted labels of every row in the loader."""
    return [CLASSES[i] for i in logits(model, loader, device, precision).argmax(-1).tolist()]


def stored_run(weighting, seed, identity=None):
    """The id of the current main run (identity.current_runs) of a weighting and seed."""
    main = camembert_runs(SELECTION, MAIN, identity)
    match = main[main["params.weighting"].eq(weighting) & main["params.seed"].eq(str(seed))]
    if match.empty:
        raise LookupError(f"no current main run of weighting {weighting} and seed {seed}")
    return match["run_id"].iloc[0]


def load_stored(run_id):
    """The model a main run stored, loaded from the store's artifacts."""
    with tempfile.TemporaryDirectory() as directory:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model", dst_path=directory
        )
        return AutoModelForSequenceClassification.from_pretrained(path)


def keeps(checkpoint, macro_f1, kept):
    """Whether this epoch replaces the kept one: always for final, on a strict gain for best."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {checkpoint!r}; expected one of {CHECKPOINTS}")
    return checkpoint == "final" or kept is None or macro_f1 > kept["macro_f1"]


def fit(model, tokenizer, train, validation, weighting, learning_rate, settings, device):
    """Train for every epoch, scoring validation after each; the kept epoch and its predictions.

    The schedule runs over all epochs. With checkpoint "best" the model ends in the state of
    the epoch with the highest validation macro-F1 (a tie keeps the earlier one).
    """
    train_loader = batches(tokenizer, train, settings, shuffle=True)
    validation_loader = batches(tokenizer, validation, settings)
    steps = settings["epochs"] * len(train_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(settings["adam_betas"]),
        eps=settings["adam_epsilon"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = get_scheduler(
        settings["schedule"], optimizer, round(settings["warmup_ratio"] * steps), steps
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=mixed_precision(device, settings["precision"])
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights(train["label"], weighting, device))
    kept = None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with autocast(device, settings["precision"]):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        predictions = predict(model, validation_loader, device, settings["precision"])
        summary, _ = scores(validation["label"], predictions)
        mlflow.log_metrics(
            {
                "train_loss": total_loss / len(train_loader),
                "val_macro_f1": summary["macro_f1"],
                "val_accuracy": summary["accuracy"],
            },
            step=epoch,
        )
        if keeps(settings["checkpoint"], summary["macro_f1"], kept):
            kept = {"epoch": epoch, "macro_f1": summary["macro_f1"], "predictions": predictions}
            if settings["checkpoint"] == "best":
                kept["state"] = {k: v.to("cpu", copy=True) for k, v in model.state_dict().items()}
    if settings["checkpoint"] == "best":
        model.load_state_dict(kept["state"])
    return kept["epoch"], kept["predictions"]


def key(point):
    """The identity of a point, independent of its key order."""
    return tuple(sorted(point.items()))


def draw(train, point):
    """The training rows of a point: every row, or the nested draw of its fraction and seed."""
    if "fraction" not in point:
        return train
    fraction = point["fraction"]
    return train.iloc[nested_subsets(len(train), [fraction], point["seed"])[fraction]]


def run(stage, point, train, validation, model_settings, settings, run_fingerprint):
    """One logged run from a fresh pinned model at a point; main runs also store their weights."""
    set_seed(point["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_settings)
    model = load_model(model_settings).to(device)
    with start_run(run_name="-".join(map(str, ["camembert", stage, *point.values()]))):
        mlflow.log_params(
            {
                "model": "camembert",
                "revision": model_settings["revision"],
                "stage": stage,
                "train_rows": len(train),
                "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "precision": "fp16" if mixed_precision(device, settings["precision"]) else "fp32",
                "fingerprint": run_fingerprint,
            }
            | point
            | {name: settings[name] for name in LOGGED}
        )
        start = time.time()
        epoch, predictions = fit(
            model,
            tokenizer,
            train,
            validation,
            point["weighting"],
            point["learning_rate"],
            settings,
            device,
        )
        mlflow.log_metrics({"train_seconds": time.time() - start, "kept_epoch": epoch})
        summary = log_evaluation(validation["label"], predictions)
        if stage == MAIN:
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                mlflow.log_artifacts(directory, "model")
    return point | {"train_rows": len(train), "epoch": epoch} | summary


def finished(stage, run_fingerprint):
    """Points of this stage finished in the store with this fingerprint, by key, with their results.

    Runs the stage left unfinished, from an interrupted command, are deleted. Finished runs with
    another fingerprint stay in the store but are not reused; two with one point raise.
    """
    done = runs(EXPERIMENT[stage], f"params.model = 'camembert' and params.stage = '{stage}'")
    if done.empty:
        return {}
    for run_id in done.loc[done["status"] != "FINISHED", "run_id"]:
        mlflow.delete_run(run_id)
    same = done.reindex(columns=["params.fingerprint"])["params.fingerprint"].eq(run_fingerprint)
    results = {}
    for _, r in done[done["status"].eq("FINISHED") & same].iterrows():
        point = {
            name: parse(r[f"params.{name}"])
            for name, parse in POINT.items()
            if pd.notna(r.get(f"params.{name}"))
        }
        if key(point) in results:
            raise ValueError(f"two finished {stage} runs share the point {point}")
        results[key(point)] = point | {
            "train_rows": int(r["params.train_rows"]),
            "epoch": int(r["metrics.kept_epoch"]),
            "macro_f1": r["metrics.macro_f1"],
            "accuracy": r["metrics.accuracy"],
        }
    return results


def run_points(stage, points, texts=None, settings=None):
    """Run every point not yet finished in the store, each on its training rows (see draw)."""
    config = load_config()
    settings = settings or config["training"]
    train, validation = train_validation(load_texts() if texts is None else texts)
    set_experiment(EXPERIMENT[stage])
    model_settings = config["models"]["primary"]
    run_fingerprint = fingerprint(settings, model_settings, train, validation)
    done = finished(stage, run_fingerprint)
    results = [
        done.get(key(point))
        or run(
            stage, point, draw(train, point), validation, model_settings, settings, run_fingerprint
        )
        for point in points
    ]
    return pd.DataFrame(results)


def run_pilot(texts=None, settings=None):
    """One run per pilot learning rate on the first seed without weighting."""
    settings = settings or load_config()["training"]
    points = [
        {"weighting": "none", "seed": settings["seeds"][0], "learning_rate": lr}
        for lr in settings["pilot_learning_rates"]
    ]
    return run_points(PILOT, points, texts, settings)


def run_main(texts=None, settings=None):
    """Weighting options x seeds at the learning rate frozen from the pilot."""
    settings = settings or load_config()["training"]
    if settings["learning_rate"] is None:
        raise ValueError("training.learning_rate is null: freeze it from the pilot first")
    points = [
        {"weighting": w, "seed": seed, "learning_rate": settings["learning_rate"]}
        for w in OPTIONS
        for seed in settings["seeds"]
    ]
    return run_points(MAIN, points, texts, settings)


def run_curves(texts=None, settings=None, fractions=None):
    """The selected weighting at the frozen learning rate on every draw below the full set.

    Nothing is retuned; the seed draws the subset and seeds the training. The full set's
    points are the main runs of that weighting.
    """
    config = load_config()
    settings = settings or config["training"]
    fractions = fractions or config["selection"]["learning_curves"]["fractions"]
    weighting = selected(current_identity(config | {"training": settings}, texts))
    learning_rate = settings["learning_rate"]
    points = [
        {"weighting": weighting, "seed": seed, "learning_rate": learning_rate, "fraction": f}
        for f, seed in draws([f for f in fractions if f < 1], settings["seeds"])
    ]
    return run_points(CURVES, points, texts, settings)


def best_weighting(main_runs):
    """The weighting with the best mean macro-F1 over seeds; a tie keeps the earlier option."""
    means = main_runs.groupby("params.weighting")["metrics.macro_f1"].mean()
    return means.reindex(OPTIONS).idxmax()


def selected(identity=None):
    """The weighting selected from the current main CamemBERT runs in the project store."""
    return best_weighting(camembert_runs(SELECTION, MAIN, identity))
