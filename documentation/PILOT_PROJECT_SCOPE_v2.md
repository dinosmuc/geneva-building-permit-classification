# Pilot project scope

## 1. Project objective

The project develops and evaluates a supervised NLP system for classifying French building-permit descriptions from the canton of Geneva. Its central question is how effectively classical text classification and a pretrained French language model can reproduce administrative operation categories from short descriptions alone.

The project is a domain-specific multiclass classification pilot. Its scope covers the dataset, text preparation, model comparison, learning behaviour, evaluation and application to new descriptions.

## 2. Dataset and target categories

The source snapshot contains 197,224 administrative records. There are 128,806 usable descriptions: 49,746 have one of the eight target labels, 79,022 are unlabelled, and 38 belong to the additional AM category.

| Category | Meaning within the study |
|---|---|
| AFF | Significant change of use |
| AGR | Additional floor area for housing or activities |
| AGV | Additional residential floor area in a villa |
| EQU | Infrastructure and public-domain improvements |
| NOU | Significant new construction or an additional building |
| PI | Minor construction in the private domain |
| REN | Maintenance or renovation without qualifying additional floor area or significant change of use |
| VIL | New villa or a major villa extension creating an additional dwelling |

These are concise descriptions of the administrative categories. AM is a valid category concerning spatial planning and urban development, but it is outside the eight-class study. Its small number of examples and absence from the selected test periods limit its evaluation under this design.

The model input is exclusively `DESCRIPTION`; the target is `TYPE_OPERATION`. Dates, identifiers, status, source and coordinates describe the dataset and support its separation, but are not predictive inputs. `OPERATION`, which describes the existing category, is also excluded from the input.

## 3. Data characteristics and preparation

Descriptions have a median length of 51 characters and a maximum of 254. Some contain spelling errors, incomplete wording, several operations or apparent truncation. Class frequencies are unequal, and identical descriptions sometimes carry different administrative labels.

Preparation distinguishes substantive descriptions from missing values and placeholders, accounts for the observed date formats and retains stable record and dossier identities. Short, ambiguous and potentially truncated descriptions remain part of the study. Administrative labels serve as reference annotations without being assumed perfectly correct.

Normalized text supports overlap analysis, while the original wording remains available for modelling. The partition design accounts for related dossiers, repeated normalized descriptions and exposure during the earlier qualitative review.

## 4. Training, validation and test scope

| Partition | Filing period | Retained records | Role |
|---|---|---:|---|
| Training | Through 2010 | 30,782 | Model fitting and learning curves |
| Validation | 2011–2012 | 2,794 | Model settings and development comparisons |
| Historical test | 2013 | 1,320 | Final pilot evaluation |
| Recent test | 2014–2026 | 6,671 | Reserved for the research extension |
| Unlabelled pool | 2014–2026 | 58,482 | Reserved for later annotation |

The counts reflect the retained partitions from data exploration; 2026 is a partial year. Temporal separation exposes differences between older and later records rather than randomly mixing all years. The pilot evaluation concerns the historical test, while the recent test remains separate from this phase's model selection.

## 5. Models and experimental comparison

The models are a majority-class baseline, TF-IDF with LinearSVC, and CamemBERTav2-base (`almanach/camembertav2-base`). The baseline establishes the performance of always choosing the most frequent training category. LinearSVC represents a classical text-classification approach, and CamemBERTav2 provides contextual language representations adapted to the target categories.

The classical comparison includes minimal preprocessing and a variant with French stop-word removal and lemmatization. CamemBERTav2 uses the original descriptions with its WordPiece tokenizer. A 128-token input limit covers the descriptions observed in the tokenization audit.

Learning curves at 10%, 30% and 100% of the training data examine sensitivity to dataset size. Transformer results across seeds 13, 42 and 73 describe variation between repeated runs. Model settings are selected using validation performance.

## 6. Evaluation and interpretation

Macro-F1 is the primary metric because the categories are imbalanced. Accuracy, per-category precision, recall and F1, class support and confusion matrices provide complementary views. Error analysis considers rare categories, ambiguous descriptions, label inconsistencies and possible truncation.

The results measure agreement with administrative labels in the defined evaluation population. They do not establish expert-level understanding of complete permit dossiers, approval prediction or performance on unrelated document domains.

## 7. Outcome and connection to SwissText

The outcome is an evaluated eight-category classifier with prediction support for new Geneva descriptions, accompanied by reproducible code, saved models, experiment results and interpretable analyses.

The pilot project forms the shared foundation for the SwissText extension. Its data preparation, label definitions, modelling and evaluation remain reusable. LLM-generated training labels and the effect of their ordering enter only in the research extension.
