#!/bin/sh
# Uploads public folder to S3
aws s3 cp checkpoints/training_run/actor/OmahaActorFinal s3://pokerai/production/OmahaActorFinal --acl public-read --cache-control max-age=0
aws s3 cp checkpoints/training_run/critic/OmahaCriticFinal s3://pokerai/production/OmahaCriticFinal --acl public-read --cache-control max-age=0