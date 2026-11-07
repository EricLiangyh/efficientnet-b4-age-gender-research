# Contributing research changes

This is currently a private research repository with no selected license. Discuss reuse or contributions with the owner before redistributing its contents.

For an experiment, record the data manifest SHA256, preprocessing version, seed, configuration, environment, checkpoint hash and evaluation partition. Keep validation and final test roles separate. Do not commit datasets, identity metadata, face photographs, credentials or local output directories.

Run `python -m unittest discover -s tests -v` and verify that any reported table is regenerated from committed logs. Document new results as new experiments; never overwrite historical evidence to match a preferred number.
