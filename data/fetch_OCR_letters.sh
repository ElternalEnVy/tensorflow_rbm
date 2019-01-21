#!/usr/bin/env bash
wget 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'
gunzip 'letter.data.gz'
mkdir -p 'OCR_letters'
mv letter.data OCR_letters
rm -f ./wget-log