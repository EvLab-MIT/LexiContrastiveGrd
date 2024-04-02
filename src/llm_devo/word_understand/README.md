# Context-based Word Understanding

Example command to evaluate a pretrained model like GPT2 taken from huggingface is:
```
python eval_word_norm.py --pretrained "gpt2" --high_level_task [task]
```

There are three choices for the `high_level_task` parameter corresponding to three benchmarks separately for Nouns (`pair_sent`, the default parameter), Verbs (`verb_pair_sent`), and Adjectives (`adj_pair_sent`).


Running the following commands will get the result.
```
>>> from llm_devo.notebooks.word_understand_metrics import WordUnderstandMetrics
>>> test = WordUnderstandMetrics(model_name='pretrained/gpt2', CACHE_DICT={}, task_name='pair_sent')
>>> test.get_aggre_perf()
(0.9460872549767371, None)
```

Masked models are currently **NOT** supported for this benchmark.
