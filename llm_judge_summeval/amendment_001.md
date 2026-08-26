# Protocol amendment 001 — provider compatibility recovery

Status: **approved after infrastructure outcomes were observed, before any
held-out human labels or quality metrics were inspected** on 2026-08-26.

The locked preregistration remains unchanged. This amendment applies only to
recovering prespecified secondary API calls that could not be executed as
written:

1. GPT-5-mini rejected the `temperature` request field as unsupported. The
   rerun omits that field and otherwise preserves the pinned model, prompt,
   schema, dataset, and analysis. This secondary cross-provider result is
   labeled amended and cannot determine primary success.
2. Claude pairwise calls returning the provider's generic HTTP 400 "Invalid
   request data" receive exactly one additional schema-identical attempt.
   Successful original responses are reused. A request that already exhausted
   malformed-output validation is not retried.
3. Original append-only ledgers are retained. Final provenance and validity
   rates include all attempts, while the latest terminal outcome per
   content-addressed request determines analyzability.
4. The confirmatory Claude pointwise prompt, model, data, estimands, thresholds,
   and outcomes are not changed.

The amendment was explicitly approved by the repository owner in the active
research session. It was motivated by provider compatibility and coverage, not
observed quality metrics.
