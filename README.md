# Phishing Email Generator

A generator of phishing emails obtained by finetuning a large language model on a publicly available set of emails.

## Project Structure

1. [Data](https://github.com/emaroppo/phish-gen/tree/main/data): load files, clean data, add features, export to formats compatible with various libraries 
2. [Finetuning](https://github.com/emaroppo/phish-gen/tree/main/prompt_generation): supervised finetuning, alignment using direct preference optimization, finetuning on human feedback using Kahneman-Tversky (binary) optimization
3. [Inference](https://github.com/emaroppo/phish-gen/tree/main/inference): use trained model to generate output

