#!/bin/sh
# Uploads public folder to S3
aws s3 cp /checkpoints/training_run/actor/OmahaActorFinal s3://pokerai/actor/OmahaActorFinal --acl public-read --cache-control max-age=0
aws s3 cp /checkpoints/training_run/critic/OmahaCriticFinal s3://pokerai/critic/OmahaCriticFinal --acl public-read --cache-control max-age=0