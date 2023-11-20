#!/usr/bin/env bash

for i in $(ls -d wandb/run*); do
    wandb sync --append ${i}
done
