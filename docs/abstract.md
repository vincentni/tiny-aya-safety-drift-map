# Abstract

**Tiny Aya Global gives a harmful answer to 9% of harmful requests. Five sentences in the system prompt bring that to 5%; overfitting a benign fine-tune takes it to 14%.**

Multilingual safety is often one number judged in English. This project maps where Tiny Aya Global gives harmful answers on 600 red-teaming prompts in six languages, by harm category and prompt condition. One closed judge labels every answer under a public rubric; checked against human labels, it is strict, so the rates are conservative.

With its standard prompt the base model gives a harmful answer to 55 of 598 prompts (9.2%): 2 of 100 in French, 16 of 99 in Russian, 16.9% on representation harms.

A five-sentence system-prompt patch on biased content and self-harm brings that to 4.8% (p < 0.001) on the prompts it was written against. It costs over-refusal (10.4% → 14.0%) and task score (-4.4 chrF).

The same 1,272 benign rows are LoRA fine-tuned twice. The light one costs no safety, though among complete answers the model is not safer, and the patch adds nothing to it (6.5% → 6.2%). The overfit one erodes it: 9.2% → 14.0% (p < 0.001), and 10.2% → 14.2% with no safety preamble (p = 0.007), each under the 5-point bar set in advance. Mixing a quarter of the base weights back (SafeTune weight interpolation) brings the latter to 8.5% (p < 0.001) and keeps the task gain.

Typed in Latin script, Hindi, Russian and Arabic prompts are refused far less (25% against 69%); the base model's harmful answers do not rise, the fine-tuned model's do (5.8% → 14.7%, p < 0.001). 4-bit quantization of the fine-tuned model changes nothing measurable.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
