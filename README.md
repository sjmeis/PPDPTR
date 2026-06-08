# Post-processing DP Text Rewriting (PPDPTR)
Code repository for the ARES IWAPS 2026 paper: *Improving the Utility, Privacy, and Human Acceptance of Differentially Private Rewritten Texts*

## Getting Started
In `PPDPTR.py`, you will find the runable class code to post-process DP rewritten texts.

Simply run:

```python
import PPDPTR
X = PPDPTR.PP(hf_token=TOKEN)
output = X.postprocess(ORIGINAL_TEXT, DP_TEXT, use_advanced_features=True)
```

`use_advanced_features` is set to activate the *advanced* variant as introduced in the paper.

The `PP` class also includes a `postprocess_batch` for batch processing.

## Datasets
All utilized datasets in the paper are included in the `data` directory.

## Citation
Please consider citing our work if you find this code useful:

```
Coming soon!
```

