#!/bin/bash

DATA_DIR=$1

for TEST_DIR in $(ls $DATA_DIR); do
    if [ -f $DATA_DIR/$TEST_DIR ]; then
	    continue
    fi
    
    for NESTED_DIR in $(ls $DATA_DIR/$TEST_DIR); do
	    cd $DATA_DIR/$TEST_DIR/$NESTED_DIR
	    bash get.sh
	    cd -
    done
done
