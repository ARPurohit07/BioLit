# Vision-model table transcription: tried, measured worse, left off

**Why.** 59% of extracted table chunks (66 of 112) had empty cells, and value questions on those tables were answered correctly
57% of the time (8 of 14) against 86% (12 of 14) when the table was clean.

**What.** `scripts/transcribe_tables.py` rewrites each table from a crop of its page with `gemma4:31b-cloud`, and accepts a
transcription only if every number in it also appears in the raw text of that page region. 86 of 101 tables were accepted.

**Result** (24 table questions in the tuning and held-out halves, same prompt, same generator, same judge, only the index changed):

| | Before | After |
|---|---|---|
| Value present in the answer | 19 | 15 (gained 0, lost 4) |
| "Insufficient evidence" answers | 3 | 6 |
| Faithfulness | 0.870 | 0.792 |

**Why it failed.** The check proves a number exists on the page, not that it sits in the right cell. `q091` went from the correct
0.9621 to 0.9721, a value from a neighbouring cell. Presence is necessary but not sufficient for a safe rewrite.

**Status.** Off by default (`chunking.use_table_transcriptions: false`). A safe version would need a cell-level check, for example
comparing each transcribed row with the words on that row of the page.
