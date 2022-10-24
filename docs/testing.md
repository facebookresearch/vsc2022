# Running tests

Run tests using the `unittest` module:

```
$ python -m unittest discover
..ss.................
----------------------------------------------------------------------
Ran 21 tests in 0.060s

OK (skipped=2)
```

The skipped tests are localization tests that only run if VCSL is installed.

If VCSL is installed (see [installation](installation.md)), no tests should be skipped.

(When run, localization tests warn about unclosed multiprocessing pools.)
