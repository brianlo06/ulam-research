#!/bin/bash
L=4000000
mkdir -p data/fam
for pair in "2 3" "2 5" "2 7" "2 9" "2 11" "2 13" "3 4" "3 5" "3 7" "3 8" "3 10" "3 11" "4 5" "4 7" "4 9" "4 11" "5 6" "5 7" "5 8" "5 9" "5 11" "5 12" "6 7" "6 11" "7 8" "7 9" "7 10" "7 12" "9 10" "10 11"; do
  set -- $pair
  f="data/fam/u${1}_${2}.bin"
  [ -s "$f" ] && continue
  ./src/ulam $1 $2 $L "$f" 2>> data/fam/scan.log
done
echo DONE_SCAN2 >> data/fam/scan.log
