# CircuitKIT check of the SQuAD probe

The check runs in an environment of its own: `pip install --no-deps -r tools/circuitkit/requirements-circuitkit.txt` into a copy of the project's development environment. The package and its tests do not import it.

    python -I tools/circuitkit/circuitkit_check.py --package src states --model MODEL_DIR --manifest EVAL/manifest.json --input OUT/squad_dev_in.jsonl --stored ACTS/squad_dev --probe OUT/probe_squad --every 5 --workers 2 --threads 10 --out CHECK
    python -I tools/circuitkit/circuitkit_check.py --package src sweep --train OUT/squad_train_in.jsonl --answers OUT/squad_train_base.jsonl --stored ACTS/squad_train --fit OUT/probe_squad_fit.json --out CHECK
    python -I tools/circuitkit/circuitkit_check.py --package src report --out CHECK --results RESULTS
