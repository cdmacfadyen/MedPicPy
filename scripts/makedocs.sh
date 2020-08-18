#!/bin/bash
pdoc medpicpy --html -o temp-docs-dir --template-dir ./scripts/pdoc-templates
mv -f temp-docs-dir/medpicpy/* docs/
rm -rf temp-docs-dir
