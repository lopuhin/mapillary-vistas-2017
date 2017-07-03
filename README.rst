mapillary-vistas-2017
=====================

Put data to ``data``::

    data
    ├── config.json
    ├── testing
    │   └── images
    ├── training
    │   ├── images
    │   ├── instances
    │   └── labels
    └── validation
        ├── images
        ├── instances
        └── labels

Use Python 3.5, run::

    pip install -r requirements.txt
    mkdir runs


Train UNet FTW::

    ./train.py runs/debug --limit 500 --batch-size 2

