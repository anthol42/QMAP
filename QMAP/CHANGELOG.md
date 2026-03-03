# Changelog
## 0.1.1 - 2026-02-03
- train_test_split now uses a rust accelerated version of the filter_out function, which is much faster for large datasets. This should not change the results, but it may change the order of the sequences in the train and test sets, and multi-threaded. It is a drop-in replacement.

