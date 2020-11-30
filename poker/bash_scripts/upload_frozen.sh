#!/bin/sh
# Uploads public folder to S3
aws s3 cp ../hand_recognition/checkpoints/multiclass_categorization/HandRankClassificationFC s3://pokerai/frozen/hand_board_weights --acl public-read --cache-control max-age=0