# Abstract

**Tiny Aya Global answers 97% of questions its passage cannot answer; a probe on its activations cuts wrong answers from 56% to 29%, 8 points better than one instruction line (36%), while 80% of answerable questions are still answered.**

A small multilingual assistant is asked questions about passages that often do not hold the answer. Answering anyway is a hallucination in the plain sense: an answer with no ground in the given text. This study trains a probe on Tiny Aya Global's internal activations to flag the answers it is about to get wrong, makes the model abstain where the probe fires, and sets the result beside what a prompt alone can do.

The test is 1,000 questions of the official SQuAD 2.0 validation file, each on its own passage: 500 that the passage cannot answer and 500 that it can. The base model answers 485 of the 500 unanswerable ones and is wrong on 560 of 1,000.

The probe is a logistic regression on the layer-20 hidden state at the last prompt token, trained on the model's own errors on 2,000 questions of the train split. Layer, regularisation and threshold are fixed on a dev split by article, and the probe is scored once. Abstaining where it fires takes wrong answers to 286 (-27.4 points, 95% CI over articles -32.3 to -22.8; 35 of 35 articles better), 505 → 248 as a closed judge reads the answers; 401 of the 500 answerable questions are still answered. No weight changes. A second sample of 1,000 questions, mostly on the same passages, gives 575 → 315.

One instruction line in the prompt, with no probe, already takes wrong answers to 364: about two thirds of the reduction for free. The probe is 7.8 points better than it (95% CI 3.89 to 11.9), one wording against one probe, and combining the two does not lower wrong answers further.

The signal is in the activations: passage length and TF-IDF of the prompt predict the model's errors with AUC 0.4978 and 0.5545, the probe with 0.8246; on the second sample the model's own output confidence reaches 0.6951 at best, the probe 0.7853. Served through a forward hook during generation, the gate gives the offline gate's answers on 100 of 100 dev prompts. CuratorKIT gates the questions, AuditKIT generates the scored answers, and CircuitKIT reads the same hidden states and finds the same layer.

What does not carry: probes trained on BBQ question pairs, a second kind of unanswerable question, barely move SQuAD (560 → 502 and 549), and a fine-tune that removes the behaviour on pairs still answers 467 of the 500 unanswerable SQuAD questions and is wrong on 529 of 1,000.

Limits: English only; one probe, one seed, one prompt wording; word-match scoring beside a judge's reading; the model's errors here are mostly answered unanswerable questions, so the probe is not shown to detect wrong answers to answerable ones.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
