#!/bin/zsh
# Parallel ranged download of Synthbuster (Zenodo 10.5281/zenodo.10066460, CC BY-NC-SA 4.0), then md5 check.
D=~/anchor_data/synthbuster_dl; cd $D
URL="https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
SIZE=12372557226; N=6; CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( (i+1)*CH - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( until [ -f part$i ] && [ $(stat -f %z part$i) -eq $((e-s+1)) ]; do
      have=$([ -f part$i ] && stat -f %z part$i || echo 0)
      curl -sL -A "research-client/1.0" -r $((s+have))-$e "$URL" >> part$i || sleep 5
    done ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo part$i; done) > synthbuster.zip
md5 -q synthbuster.zip > md5.txt
if [ "$(cat md5.txt)" = "0695bd328e16ea21c5c9cc2ae1d994ff" ]; then echo "MD5 OK" >> md5.txt; rm part*; else echo "MD5 MISMATCH" >> md5.txt; fi
echo "DOWNLOAD FINISHED $(date)" >> md5.txt
