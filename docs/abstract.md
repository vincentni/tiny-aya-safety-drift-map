# Abstract

**Tiny Aya Global gives a harmful answer to 9% of harmful requests. Five sentences in the system prompt bring that to 5%; overfitting a benign fine-tune takes it from 10% to 14%.**

Multilingual safety is usually reported as one number judged in English. This project maps where Tiny Aya Global (3.35B) gives harmful answers on 600 red-teaming prompts in six languages, by language, harm category and prompt condition. One closed judge labels every answer under a public rubric; against human labels it matches on refusals (kappa 0.93 to 1.00) and is strict on harmful answers, so the rates are conservative.

With its standard prompt the base model gives a harmful answer to 55 of 598 prompts (9.2%): 16 of 99 in Russian against 2 of 100 in French, and 16.9% on representation harms.

A five-sentence system-prompt patch on biased content and self-harm brings that to 4.8% (-4.3 points, p < 0.001). It is paid for with over-refusal (10.4% → 14.0%) and task score (-4.4 chrF). The safety paragraph the model ships with adds refusals but no measurable safety.

The same 1,272 benign multilingual rows are LoRA fine-tuned under two recipes. The light one (3 epochs, $0.49) costs no safety, though among complete answers the model is not safer. The overfit one (10 epochs, $1.45), measured with no safety preamble, erodes it: 10.2% → 14.2% (p = 0.007), mostly in Arabic and Russian, under the 5-point bar set in advance for calling a checkpoint drifted.

4-bit quantization of the fine-tuned model brings no measurable change. Hindi, Russian and Arabic prompts typed in Latin script are refused far less often (25% against 69%) with no measurable rise in harmful answers.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
