Code for both ["Visual Grounding Helps Learn Word Meanings in Low-Data Regimes"](https://arxiv.org/abs/2310.13257) and ["Lexicon-Level Contrastive Visual-Grounding Improves Language Modeling"](https://arxiv.org/abs/2403.14551)

# Environment

Python 3.9, transformer package in huggingface, and datasets package in huggingface.

Evaluation requires `lm_eval` repo ([link](https://github.com/chengxuz/lm-evaluation-harness.git)).

Install this repo via `pip install -e .`


# Evaluation

## Word-Relatedness

See the README file in the `./src/llm_devo/word_sim/` folder.

## Semantic-Feature Prediction

See the README file in the `./src/llm_devo/word_norm/` folder.

## Lexical-Relation Prediction

See the README file in the `./src/llm_devo/lexical_relation/` folder.

## Part-Of-Speech Prediction

See the README file in the `./src/llm_devo/pos_pred/` folder.
