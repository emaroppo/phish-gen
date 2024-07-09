# Phishing Email Generator

A generator of phishing emails obtained by finetuning a large language model on a publicly available set of emails. It is based around 3 components:
1. [Offline finetuning](https://github.com/emaroppo/phish-gen/tree/main/offline_finetuning/train): train a model that can generate phishing emails matching the prompt submitted by a user.
2. [Prompt generator](https://github.com/emaroppo/phish-gen/tree/main/prompt_generator): test different templates and apply them consistently to the fine-tuning dataset, as well as user input.
3. [Inference](https://github.com/emaroppo/phish-gen/tree/main/inference): use a fine-tuned model to generate the messages
