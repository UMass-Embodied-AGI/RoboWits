#!/usr/bin/env bash
# Apply local patches to the installed genesis package.
# Run after: pip install genesis  or  uv sync
set -e

SITE=$(python -c "import site; print(site.getsitepackages()[0])")

echo "Applying patches to $SITE ..."

for patch in "$(dirname "$0")"/*.patch; do
    echo "  -> $patch"
    patch -d "$SITE" -p0 --forward --reject-file=- < "$patch"
done

echo "Done."
