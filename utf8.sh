#!/bin/sh

for i in pos/*.txt
do
    iconv -f GB2312 -t utf-8 $i > new/$i
done

for i in neg/*.txt
do
    iconv -f GB2312 -t utf-8 $i > new/$i
done
