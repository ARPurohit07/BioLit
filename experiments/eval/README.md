# experiments/eval

Committed evidence behind the numbers in the top-level README (§9). Everything here is produced by a script in `scripts/`; nothing is edited by hand.

## Question sets
| File | What it is |
|---|---|
| `eval_set.jsonl` | The 144 generated questions, references unchecked |
| `eval_audit.json`, `eval_set_clean.jsonl` | Deterministic checks on that set (superseded by the model audit) |
| `reference_verification.jsonl`, `eval_set_audited.jsonl` | A large model checked each reference against its source chunk; 130 kept |
| `eval_set_final.jsonl` | The 126 used for every current number |
| `eval_set_dev.jsonl` / `eval_set_test.jsonl` | 64 to tune on / 62 held out |
| `eval_set_remapped_semantic.jsonl`, `chunk_metadata_old.jsonl` | Labels re-pointed onto semantic chunks, for that experiment (§9.4) |

## Served by the Evaluation page
`retrieval_eval.json` (`scripts/eval_retrieval.py`) and `ragas_summary.json` (`scripts/publish_eval.py`). `ragas_summary_3b_judge.json` and `retrieval_eval_144_unaudited.json` are the earlier versions, kept for the record.

## Answer quality
| File | What it is |
|---|---|
| `final_result_cloud.json`, `scores_final_cloud.jsonl`, `answers_final.jsonl` | 130 answers judged by the 120B model; per-question scores |
| `final_result_local.json`, `scores_final_local.jsonl`, `final_result.json`, `scores_final.jsonl` | The same run scored by the 3B judge (not validated) |
| `judge_controls_cloud.json`, `judge_controls.json`, `judge_controls_adapted.json` | Right-versus-wrong controls per judge |
| `pr_*.jsonl`, `faith_*.jsonl` | Per-question claim precision/recall and faithfulness for the tuning (`dev`) and held-out (`test`) halves; `_v2` is the scope-matching prompt, `_hf` High-Faithfulness mode, `_tables_new` the table-transcription index |
| `diagnosis_dev*.json`, `diagnosis_test_v2.json` | Why low-scoring answers lost points |
| `ab_*`, `prompt_ab.json` | The earlier prompt A/B on 29 questions (§9.2), including the runs that exposed the marker bug |

## Retrieval and chunking experiments
`retrieval_sweep.json` (fusion weights, candidate pool, depth), `retrieval_eval_semantic*.json` and `retrieval_eval_*baseline*/fixed_gn/audited.json` (semantic chunking, §9.4), `table_transcription.md` (vision-model tables, §9.4).
