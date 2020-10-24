#!/bin/sh
aws s3 sync s3://pokerai/production checkpoints/production --no-sign-request