#!/bin/sh  
#============ get the file name ===========  
Folder_A="/home/mc_zilla/data/raw_data/test_data/"  
for file_a in ${Folder_A}/*
do  
    temp_file=`basename $file_a`  
    c=$Folder_A$temp_file
    ./copmute_feature -all $c
done