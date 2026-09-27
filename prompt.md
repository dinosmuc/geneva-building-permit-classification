# Annotation prompt

**Status: not written.** Blocked on the official definitions of the eight operation
categories (AFF, AGR, AGV, EQU, NOU, PI, REN, VIL). The prompt must quote an
authoritative source, not a paraphrase, so this waits on the SITG metadata.

Once unblocked, this file holds the frozen French prompt and nothing else. It is
committed before bulk annotation and never edited afterwards; a change means a new
prompt file and a new annotation run.

Required structured output per record:

| Field | Values |
|---|---|
| `category` | one of the eight codes, or `AM`, or `insufficient_evidence`, or `abstain` |
| `span` | the substring of the description supporting the choice |
| `status` | `ok` \| `insufficient_evidence` \| `out_of_scope` \| `abstain` |

Acceptance rules are frozen before bulk annotation. Self-reported confidence is not a
calibrated probability. Disagreement with the recorded code is not by itself teacher error.
