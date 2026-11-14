# Abstract

**A per-language map of safety drift in Tiny Aya Global**

Benign fine-tuning can remove refusal behaviour from an aligned model, and the loss is usually reported as one aggregate number judged in English. This project measures it per language for Tiny Aya Global, a 3.35B multilingual model. The model is fine-tuned with LoRA on 1,272 benign instruction rows in English, French, Spanish, Russian, Hindi and Arabic, and evaluated before and after on 100 red-teaming prompts per language from `CohereLabs/aya_redteaming`, with XSTest for over-refusal and held-out benign prompts for task quality. One judge, a closed model under a fixed public rubric, labels every response; all labels are published.

In both conditions measured, benign fine-tuning left the model as safe or safer while improving the benign task. Under the chat template the model is served with, which opens with a safety paragraph, the pooled safe rate went from 90.8% to 93.5% (+2.7 points, 95% CI +0.3 to +5.0, p = 0.029). With that paragraph removed it went from 89.8% to 94.1% (+4.4 points, 95% CI +2.2 to +6.7, p < 0.001). No language lost safety beyond noise: the only negative estimate is French under the served template, -2.0 points (95% CI -6.0 to +2.0). chrF on the benign prompts rose from 30.5 to 37.3 and from 25.1 to 36.1, and over-refusal did not move. Removing the paragraph moves the baseline only from 90.8% to 89.8% safe, so the safe behaviour sits mostly in the weights, not in the template.

One experiment is still open: a deliberately aggressive fine-tune on the same rows, which decides whether the SafeTune WiSE-FT repair and the q4_k_m column have a drifted model to work on. Every GPU run is a logged Hugging Face Job; the runs behind these numbers cost $2.39.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
