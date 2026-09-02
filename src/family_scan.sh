#!/bin/bash
# generate a grid of Ulam-type sequences U(u,v)
L=4000000
mkdir -p data/fam
for u in 1 2 3 4 5; do
  for v in $(seq $((u+1)) 60); do
    f="data/fam/u${u}_${v}.bin"
    [ -s "$f" ] && continue
    ./src/ulam $u $v $L "$f" 2>> data/fam/scan.log
  done
done
echo DONE_SCAN >> data/fam/scan.log
