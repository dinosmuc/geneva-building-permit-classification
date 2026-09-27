# Geneva building-permit classification

This project tests whether the order in which a compact neural classifier sees LLM-generated
labels and original administrative labels changes what it learns. Free-text French descriptions
from Geneva's open building-permit register are mapped to one of eight recorded operation
categories, using the description field alone. A teacher model annotates real unlabelled
permits from the recent filing period, and four training schedules — original labels only, the
two sources interleaved, pseudo-labels then originals, and originals then pseudo-labels — are
compared at a matched update budget across three seeds, against majority-class and TF-IDF
baselines. The aim is retrospective completion of missing administrative categories for search
and human review, not prediction of permit approval; a null result is a valid outcome.
