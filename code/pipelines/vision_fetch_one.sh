#!/bin/zsh
u="$1"; p="${u#https://lesc.dinfo.unifi.it/VISION/dataset/}"
[ -s "$p" ] && exit 0
mkdir -p "${p:h}"; curl -sSfL -A "research-client/1.0" --retry 3 -o "$p" "$u" || echo "FAILED $u"
