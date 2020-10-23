#!/bin/sh
# Uploads public folder to S3
aws s3 cp ../hand_recognition/checkpoints/multiclass_categorization/HandRankClassificationFive s3://pokerai/frozen --acl public-read --cache-control max-age=0