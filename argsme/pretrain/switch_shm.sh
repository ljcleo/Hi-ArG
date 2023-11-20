#!/usr/bin/env bash

target=argsme/train_${1}
rm -rf /dev/shm/*

for x in $(ls /dev/shm); do
	rm /dev/shm/${x};
done

rsync --progress shm_cache/${target}/* /dev/shm
