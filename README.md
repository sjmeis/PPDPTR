# Post-processing DP Text Rewriting (PPDPTR)
This is the anonymous code repository for the ACL RR October 2025 submission: *Improving the Utility, Privacy, and Human Acceptance of Differentially Private Rewritten Texts*

## Getting Started
In `PPDPTR.py`, you will find the runable class code to post-process DP rewritten texts.

Simply run:
```
import PPDPTR
X = PPDPTR.PP(hf_token=TOKEN)
output = X.postprocess(ORIGINAL_TEXT, DP_TEXT, use_advanced_features=True)
```

`use_advanced_features` is set to activate the *advanced* variant as introduced in the paper.

The `PP` class also includes a `postprocess_batch` for batch processing.

## Datasets
All utilized datasets in the paper are included in the `data` directory.
