# Retrieval Evaluation Observations

## Scope

The evaluation set contains eight questions covering five indexed Official Journal documents. The observations below are based on the indexed chunk content and chunk metadata. Live score-based validation is still pending because the configured Qdrant API key is the literal `$QDRANT_API_KEY` placeholder and Qdrant returns `403 Forbidden`.

## Queries Likely To Work Well

- **Exchange rate for the US dollar**: The question contains the date, currency, and document topic. The exchange-rate document is a single complete chunk, so the answer should be easy to retrieve with complete context.
- **Which UN regulation replaces UN Global Technical Regulation No 24**: The brake-emissions document contains the old and new standard names in the same chunk. This is a strong exact-fact query with distinctive terms.
- **What is the business indicator a proxy for?**: The operational-risk regulation states directly that the business indicator is a financial-statement-based proxy for operational risk. The expected answer is concentrated near the beginning of the document.
- **How are business indicator components mapped?**: The implementing regulation explicitly mentions FINREP templates and corresponding cells. The document identifier and terminology are distinctive enough to support good retrieval.

## Queries That May Need Refinement

- **Forced-labour questions**: The forced-labour guidance contains 414 chunks, with 251 under 100 characters and many punctuation-only fragments. Retrieval may return incomplete or low-information chunks even when the source document is correct. Queries should include distinctive terms such as `Regulation (EU) 2024/3015`, `forced labour`, and `Union market`.
- **Income and expense categories in the interest, leases and dividend component**: This question asks for several categories and may retrieve only one relevant passage. Split it into narrower checks for interest income and expenses, leases, dividends, and the service component.
- **Vehicle categories covered by the brake-emissions standard**: The expected answer includes `M1`, `N1`, and `Euro 7ext`, but these terms may occur in a different chunk from the explanation of UN Regulation No 179. Evaluate both source hit and answer-keyword coverage.
- **Forced-labour scope**: This overlaps heavily with the forced-labour regulation question. Use it as a paraphrase test, but do not treat two hits from the same fragmented region as independent evidence of retrieval quality.

## Metadata Observations

- All current evaluation records expect the topic label `EU_Regulation`. This label does not distinguish exchange rates, forced labour, operational risk, or brake emissions, so topic alignment is not currently a useful quality signal.
- Source filename is more useful than topic label for narrowing this corpus. A source filter such as `source_filename == L_202601214EN.pdf` should produce a focused brake-emissions evaluation.
- Text previews are useful for manual inspection, but a truncated preview cannot establish context completeness. Inspect the full `original_text` when a preview ends with `...`.

## Recommended Next Checks

1. Supply a valid Qdrant API key and run `python evaluate_retrieval.py`.
2. Record Recall@1, Recall@3, and Recall@5 by expected source filename.
3. Check whether scores descend monotonically and whether the top result has the expected source and topic.
4. Manually review short or punctuation-only matches.
5. Compare semantic chunking with a fixed-size, overlapping baseline after merging short chunks.
