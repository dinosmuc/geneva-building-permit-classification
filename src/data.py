"""Load, clean, split and freeze the SIT_AUTOR_DOSSIER export.

Responsibilities:
  - read the raw SITG export and record source bytes, hash and download date
  - drop answer-bearing and procedural fields listed in config.yaml
  - deduplicate repeated dossier observations and keep dossier groups intact
  - assign the calendar partitions and write manifests to data/splits/
  - remove pool records whose folded description matches any labelled record
  - derive the text-disjoint evaluation subset against O, accepted P and validation

Normalisation is used for grouping only. Model input keeps the original text.
"""
