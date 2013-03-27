#!/bin/sh

find . -iname "*.py" -print0 | xargs -0 grep -l "xrutils"


find . -iname "*.py" -print0 | xargs -0 -n1 \
    sed -i -e 's/import xrutils\( \|$\)/import xrayutilities\1/g' \
        -e 's/from xrutils import\( \|$\)/from xrayutilities import\1/g' \
        -e 's/xrutils\./xrayutilities\./g' \
        -e 's/\(= *\)xrutils/\1xrayutilities/g'

