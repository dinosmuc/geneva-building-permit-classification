"""Teacher annotation of the unlabelled pool.

Responsibilities:
  - render the frozen prompt.md against a batch of real descriptions
  - call the teacher API with resumable, rate-limited requests
  - cache by (exact input, prompt hash, model configuration) so a rerun costs nothing
  - validate structured output against the schema and apply the frozen acceptance rules
  - record provider, model revision, date, decoding settings, tokens, cost and failures
  - write accepted annotations to data/prepared/ and publish accept/reject counts

No test example ever enters a prompt, a demonstration or the annotation policy.
"""
