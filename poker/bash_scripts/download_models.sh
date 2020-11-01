#!/bin/sh
aws s3 sync s3://pokerai/production ${PWD}/poker/checkpoints/production --no-sign-request