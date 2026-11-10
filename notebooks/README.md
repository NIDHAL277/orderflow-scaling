# Notebooks — exploration only

Notebooks explore; the package computes. No number of the report comes from a notebook:
every figure and table is rebuilt by `make report` from the stored results.

Read the results with the store rather than raw files:

```python
from ofs.store import Store
store = Store("data")
est = store.estimates("development")
```

Before the tag `prereg-v1`, do not estimate exponents on real data here either.
