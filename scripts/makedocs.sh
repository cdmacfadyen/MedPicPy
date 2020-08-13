#!/bin/bash
pdoc medpicpy --html -o temp-docs-dir
mv -f temp-docs-dir/medpicpy/* docs/
rm -rf temp-docs-dir
