# Abstract

**A per-language map of safety drift in Tiny Aya Global**

Benign fine-tuning can remove refusal behaviour from an aligned model, and the loss is usually reported as one aggregate number judged in English. This project measures it per language for Tiny Aya Global, a 3.35B multilingual model. The model is fine-tuned with LoRA on 1,272 benign instruction rows in English, French, Spanish, Russian, Hindi and Arabic, and evaluated before and after on 100 red-teaming prompts per language from `CohereLabs/aya_redteaming`, 250 XSTest prompts for over-refusal and 100 held-out benign prompts per language for task quality. One judge, a closed model under a fixed public rubric, labels every response for refusal and harm; all labels are published.

Under the chat template the model is served with, which opens with a safety paragraph, the fine-tune cost no safety. The pooled safe rate went from 90.8% to 93.5% (+2.7 points, 95% CI +0.3 to +5.0, p = 0.029, 597 prompt pairs), and no language lost safety beyond noise: the largest drop is French at -2.0 points (95% CI -6.0 to +2.0). No gap was measured between Hindi and Arabic and the four European languages. The refusal rate went from 70.6% to 74.4%, XSTest over-refusal from 10.4% to 9.2%, and chrF on held-out benign prompts from 30.5 to 37.3: the model learned the task without refusing more of what is safe.

Three columns are still open: the same measurement with the safety paragraph removed from the template, a WiSE-FT weight-interpolation repair with SafeTune, and the repaired model at q4_k_m. Every GPU run is a logged Hugging Face Job; the runs behind these numbers cost $1.57.

Code: https://github.com/vincentni/tiny-aya-safety-drift-map. Write-up: [article.md](article.md).
