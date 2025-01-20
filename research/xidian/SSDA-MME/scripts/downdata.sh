#!/bin/sh
mkdir data data/multi
cd data/multi
 wget -c http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip -O sketch.zip
wget -c http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip -O clipart.zip
wget -c http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip -O painting.zip
 wget -c http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -O real.zip

unzip real.zip
unzip sketch.zip
unzip clipart.zip
unzip painting.zip
unzip txt.zip