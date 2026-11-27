#!/bin/zsh
# E5: download the VISION subset (CC BY-SA 4.0; list <data>/e5/vision_subset.txt, seed 41). Resumable.
cd "${VISION_DIR:?set VISION_DIR}"
xargs -P 4 -n 1 "$(dirname "$0")/vision_fetch_one.sh" < ../vision_subset.txt
echo "E5 DOWNLOAD FINISHED $(find . -type f | wc -l) files"
