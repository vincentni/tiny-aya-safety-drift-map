# Abstract

**Tiny Aya Global gives a harmful answer to 1 in 6 harmful requests in Russian and 1 in 50 in French. The same benign data costs no safety under a light fine-tune and erodes it under an overfit one.**

Multilingual safety is usually reported as one number judged in English. This project maps where Tiny Aya Global (3.35B) gives harmful answers on 600 red-teaming prompts in six languages, by language, harm category and prompt condition. One closed judge labels every answer under a public rubric; against human labels it matches on refusals (kappa 0.93 to 1.00) and is strict on harmful answers, so the rates are conservative.

With its standard prompt the base model gives a harmful answer to 55 of 598 prompts (9.2%): 16 of 99 in Russian against 2 of 100 in French, and 16.9% on representation harms. The template's safety paragraph adds refusals but no measurable safety. A five-sentence system-prompt patch does: 9.2% → 4.8% (p < 0.001), paid for with over-refusal (10.4% → 14.0%) and task score.

The same 1,272 benign multilingual rows are LoRA fine-tuned under two recipes. The light one (3 epochs, $0.49) costs no safety: harmful answers 9.2% → 6.5% (p = 0.029), over-refusal unchanged, mostly because answers are shorter; among complete answers it is not safer. The overfit one (10 epochs, $1.45), measured with no safety preamble, erodes it: 10.2% → 14.2% (p = 0.007), mostly in Arabic and Russian, under the 5-point bar set in advance for calling a checkpoint drifted. In that condition the light recipe gives 10.2% → 5.9%.

4-bit quantization of the fine-tuned model costs no safety (42 against 43 harmful answers). In a pilot, Hindi in Latin script loses refusals (7 of 29 against 22).

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
