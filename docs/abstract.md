# Abstract

**Answer or Abstain: an activation probe for hallucination detection in Tiny Aya Global**

A small multilingual assistant is asked questions about passages that often do not hold the answer; answering anyway is a hallucination. This study trains a probe on Tiny Aya Global's internal activations to flag the answers it is about to get wrong, makes the model abstain where the probe fires, and sets the result beside a prompt instruction and a fine-tune.

The test is 1,000 questions of the official SQuAD 2.0 validation file: 500 that the passage cannot answer and 500 that it can. The base model answers 485 of the 500 unanswerable ones and is wrong on 560 of 1,000.

The probe is a logistic regression on the layer-20 hidden state, trained on the model's own errors on 2,000 questions of the train split. Layer, regularisation and threshold are fixed on a dev split by article, and the probe is scored once. Abstaining where it fires takes wrong answers to 286 (-27.4 points, 95% CI over articles -32.3 to -22.8), 505 → 248 as a closed judge reads the answers; 401 of the 500 answerable questions are still answered. A second sample of 1,000 questions, mostly on the same passages, gives 575 → 315.

One instruction line in the prompt already takes wrong answers to 364: about 70% of the reduction for free. The probe is 7.8 points better than it (95% CI 3.89 to 11.9), and combining the two does not lower wrong answers further.

A LoRA fine-tune on the same 2,000 training questions does better than the probe: 174 wrong answers (-11.2 points, 95% CI -14.27 to -7.94 against the probe gate) and 431 of 500 answerable questions still answered. It costs a training run, changes the weights, and was trained on the test's dataset and format; where the weights must stay fixed, the probe gate gets about 71% of the fine-tune's reduction.

The signal is in the activations: passage length and TF-IDF of the prompt predict the model's errors with AUC 0.4978 and 0.5545, the probe with 0.8246; on the second sample the model's own output confidence reaches 0.6951 at best, the probe 0.7853.

Two probes and a fine-tune trained on another kind of unanswerable question (BBQ pairs) barely move SQuAD (560 → 502, 549 and 529).

Limits: English only; one probe, one fine-tune, one seed each, one prompt wording; word-match scoring beside a judge's reading.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
