# commonsense-part-whole
Dataset for the [EMNLP 2019 paper](https://www.aclweb.org/anthology/D19-1625.pdf) 

```Do Nuclear Submarines Have Nuclear Captains? A Challenge Dataset for Commonsense Reasoning over Adjectives and Objects```

```James Mullenbach, Jonathan Gordon, Nanyun Peng and Jonathan May```

The dataset consists of part-whole relations along with an adjective describing the whole. The task is to infer the relationship between the adjectives and the parts.

- `data/` contains the dataset in lexical format (i.e. just the whole, part, and adjective words), and in NLI format (with retrieved premises and templated hypotheses).
- `mturk/` contains the prompt templates and scripts used to create the AMT tasks.
