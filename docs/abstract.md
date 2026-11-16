# Abstract

**Five sentences in the system prompt cut Tiny Aya Global's harmful answers from 8.3% to 6.2% on 600 held-out prompts in six languages (9.2% → 4.8% on the prompts they were written against).**

Multilingual safety is often one number judged in English. This project maps where Tiny Aya Global gives harmful answers on 600 red-teaming prompts in six languages, by harm category and prompt condition. One closed judge labels every answer under a public rubric; checked against human labels, it is strict, so the rates are conservative.

With its standard prompt the base model gives a harmful answer to 55 of 598 prompts (9.2%): 2 of 100 in French, 16 of 99 in Russian, 16.9% on representation harms. 52 of the 55 are among the answers cut off at the 256-token limit, so the rate is mostly one of long answers. On a second draw of 600 prompts French is 7 of 100, Russian 13 of 99 and representation harms 20.0%.

A five-sentence system-prompt patch on biased content and self-harm takes the base model's harmful answers from 50 of 599 (8.3%) to 37 of 597 (6.2%) on 600 held-out prompts, a fresh draw from the same source (-2.2 points, 95% CI -4.0 to -0.5, p = 0.029, uncorrected), against -4.3 points on the 600 prompts it was written against (in-sample, 9.2% → 4.8%); the two are not told apart, and the held-out result rests on 31 changed verdicts, 22 to safe and 9 to harmful. It costs over-refusal (10.4% → 14.0%) and task score (-4.4 chrF).

The same 1,272 benign rows are LoRA fine-tuned twice. The light one costs no safety, though among complete answers the model is not safer, and the patch adds nothing to it (6.5% → 6.2%). The overfit one erodes it: 9.2% → 14.0% (p < 0.001), and 10.2% → 14.2% with no safety preamble (p = 0.007), each under the 5-point bar set in advance. Mixing a quarter of the base weights back (SafeTune weight interpolation) brings the latter to 8.5% (p < 0.001) and keeps the task gain.

4-bit quantization of the fine-tuned model changes nothing measurable. Latin-script and code-mixed variants of the prompts, written for this study by the closed model because no official version exists, are in an appendix and are not part of the claim.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
