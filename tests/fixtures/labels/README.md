# Label fixtures

Synthetic label rows for the table tests: placeholder prompt ids, placeholder prompt text and made-up judge labels.
Nothing here comes from a real dataset or a real model run.

- `baseline/en.jsonl`, `drifted/en.jsonl`: 12 harmful and 6 XSTest rows each; the drifted file has one `unscored` row.
- `../bootstrap_600_pairs.json`: 600 `[earlier, later]` label pairs with a +1/6 shift in the safe rate.
