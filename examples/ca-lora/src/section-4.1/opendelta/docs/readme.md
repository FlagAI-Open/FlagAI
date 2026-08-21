# OpenDelta Documentation

To build this doc locally, please firstly install [sphinx](https://www.sphinx-doc.org/en/master/) packages.

```
pip install sphinx
pip install sphinx_rtd_theme
pip install sphinx_copybutton
pip install sphinx_toolbox
pip install myst_parser
```

Then install opendelta either from source, or from pip. After that,

```
cd docs
make html
```

Then open the generated `docs/build/html/index.html` in your local browser. 