# SwissText research scope

## 1. Research objective

The study investigates how the order of original administrative labels and LLM-generated labels affects classification of French Geneva building-permit descriptions. It extends the pilot project using the same description-only input, eight-category classification task and CamemBERTav2-base (`almanach/camembertav2-base`).

The central question is whether training first on LLM-labelled descriptions and then on original-labelled descriptions produces different results from mixing the two sources or reversing their order. Improvement is an empirical possibility, not an assumption.

## 2. Relationship to the pilot project

The research shares the pilot project's source data, category definitions, text preparation, dossier grouping, training implementation and evaluation framework. Its additional scope consists of LLM annotation, label-quality assessment, controlled training-order comparisons and analysis on the reserved recent test.

The pilot project's trained model is a previous experimental result. The research conditions start from a common pretrained CamemBERTav2 checkpoint, allowing their training histories to be compared from the same starting point.

## 3. Original and LLM-labelled data

Dataset O contains descriptions with original administrative labels from the training partition, currently 30,782 records filed through 2010. Dataset P contains existing unlabelled descriptions that receive accepted categories from an LLM teacher.

The current candidate pool for P contains 58,482 records filed during 2014–2026. These are real descriptions rather than generated text. The final size and class composition of P depend on annotation outcomes and any declared sampling scope.

The teacher's category decisions are based on the descriptions and defined label meanings. Assigned categories, ambiguity, insufficient information and out-of-scope cases are distinct outcomes. AM remains outside the eight-category task rather than becoming another target category or being forced into an existing one.

Teacher identity, prompt version and acceptance policy define the annotation procedure. A separately reviewed sample supports assessment of label quality and unresolved cases. The original administrative annotations are imperfect references, and the earlier AI-assisted qualitative review is distinct from independent human assessment.

## 4. Training arrangements

| Condition | Arrangement | Purpose in the comparison |
|---|---|---|
| Original only | Original-labelled examples throughout | Reference without LLM labels |
| Mixed | Interleaved original-labelled and LLM-labelled batches | Combined supervision without separate stages |
| LLM then original | LLM-labelled batches followed by original-labelled batches | Effect of finishing with administrative labels |
| Original then LLM | Original-labelled batches followed by LLM-labelled batches | Effect of the reverse order |

The combined conditions share the same O and P examples and source-exposure counts. Their difference is the ordering of batches. Common initialization, effective batch size, optimization settings and total training budget support that comparison. The original-only condition matches the total budget through greater exposure to original-labelled examples.

The primary comparison uses fixed-budget final checkpoints. Seeds 13, 42 and 73 give 12 main runs across the four conditions. The primary contrast is LLM-then-original versus mixed training; the other conditions provide the surrounding comparison.

## 5. Evaluation population and time setting

The reserved recent test currently contains 6,671 labelled records from 2014–2026, with partial coverage of 2026. Its separation accounts for related dossiers, repeated normalized descriptions and known development exposure. Validation supports development; the recent test provides the final research evaluation. The historical test has already served the pilot project and is a secondary diagnostic here.

The main setting is retrospective completion. LLM-labelled and test descriptions cover overlapping periods, and some training descriptions can postdate individual test descriptions. The study therefore concerns classification from a later data snapshot rather than a historical forecast.

A complementary filing-date analysis restricts LLM-labelled descriptions to filings through 2020 and evaluates test filings from 2021 onward. It separates the filing periods without establishing historical availability of labels or pretrained models.

## 6. Metrics and analysis

Macro-F1 on the recent test is the primary outcome. Per-category precision, recall, F1 and support show whether aggregate differences hide gains or losses in smaller classes. Repeated-seed results describe training variability, while paired comparisons and dossier-grouped uncertainty estimates describe differences between schedules.

Analysis across filing periods, sources, dossier types and description characteristics provides context for the results. Annotation coverage and human-review findings describe the additional supervision introduced by P.

## 7. Contribution and limits

The contribution is empirical evidence about training-order effects in a real administrative NLP application. Positive, negative and inconclusive effects all contribute to the answer.

Original and LLM-labelled examples differ in time, source and content as well as annotation method. Consequently, the study concerns the ordering of these particular supervision sources rather than a pure causal effect of label noise. The recent labelled test is also selective, so its scores do not establish accuracy across the entire unlabelled population.

The research outcome comprises the controlled comparison, annotation-quality evidence and reproducible experimental artifacts. Its scope is the Geneva classification study and its extension of the pilot implementation.
