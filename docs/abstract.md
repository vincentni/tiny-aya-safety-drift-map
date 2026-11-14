# Abstract

**Where a small multilingual model's safety is thin, and what a $0.49 fine-tune does about it**

Safety of a multilingual model is usually reported as one aggregate number judged in English. This project maps where Tiny Aya Global (3.35B) gives harmful answers: 600 red-teaming prompts from `CohereLabs/aya_redteaming` in six languages, broken down by language, harm category and prompt condition. One closed judge labels every answer under a fixed public rubric, and every label is published.

The base model answers 55 of 598 prompts harmfully (9.2%). The rate is 16 of 99 in Russian against 2 of 100 in French, and highest for representation harms (12 of 71) and self-harm (11 of 67). The safety paragraph in the model's chat template adds refusals (4.2 points, p = 0.005) but no measurable safety: 55 of 598 harmful answers with it, 61 of 599 without.

A LoRA fine-tune on 1,272 benign multilingual rows (37 minutes on one L4, $0.49) was followed by fewer harmful answers: 55 → 39 of 598 (p = 0.029), and 61 of 599 → 35 of 597 with the safety paragraph removed (p < 0.001). Over-refusal on XSTest did not move. The fine-tuned model answers far more briefly, and the drop sits in answers that no longer run to the token limit; among answers that end on their own it is not safer. Self-harm and representation harms barely move, and 16 of its 39 harmful answers are on prompts the base model answered safely.

Still being measured: script variants of the prompts, a system-prompt patch, a more aggressive fine-tune and human labels for the judge.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
