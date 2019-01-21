#!/usr/bin/env bash
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz'
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz'
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz'
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz'
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz'
wget 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz'
gunzip smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
gunzip smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
gunzip smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
mkdir -p 'NORB'
mv smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat NORB
mv smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat NORB
mv smallnorb-5x46789x9x18x6x2x96x96-training-info.mat NORB
mv smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat NORB
mv smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat NORB
mv smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat NORB
rm -f ./wget-log