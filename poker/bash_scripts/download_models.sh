#!/bin/sh
aws s3 sync s3://pokerai/production ${PWD}/checkpoints/production --no-sign-request