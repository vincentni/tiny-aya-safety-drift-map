# Abstract

**Tiny Aya Global gives a harmful answer to 1 in 6 harmful requests in Russian and 1 in 50 in French. A $0.49 benign fine-tune did not make it worse.**

Safety of a multilingual model is usually reported as one aggregate number judged in English. This project maps where Tiny Aya Global (3.35B) gives harmful answers: 600 red-teaming prompts from `CohereLabs/aya_redteaming` in six languages, broken down by language, harm category and prompt condition. One closed judge labels every answer under a fixed public rubric; against human labels it matches on refusals in every language (kappa 0.93 to 1.00) and is strict on harmful answers, so the rates below are conservative.

With its standard prompt the base model gives a harmful answer to 55 of 598 prompts (9.2%): 16 of 99 in Russian against 2 of 100 in French, and 16.9% on representation harms and 16.4% on self-harm. French is the safest language with and without the template's safety paragraph. That paragraph adds refusals (4.2 points, p = 0.005) but no measurable safety: 55 of 598 harmful answers with it, 61 of 599 without.

A LoRA fine-tune on 1,272 benign multilingual rows (37 minutes on one L4, $0.49) costs no safety: harmful answers go from 9.2% to 6.5% (p = 0.029) with over-refusal unchanged. That is mostly because its answers are shorter (answers cut at the 256-token limit fall from 45% to 12%) and it refuses slightly more; among complete answers it is not safer, and 16 of its 39 harmful answers are on prompts the base model answered safely. Self-harm and representation harms barely move.

Still being measured: a targeted system-prompt patch, and romanized and code-mixed variants of the prompts.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
