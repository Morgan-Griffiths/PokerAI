#!/bin/sh
aws s3 sync s3://pokerai/frozen ${PWD}/poker/checkpoints/frozen_layers --no-sign-request
# aws s3 cp s3://pokerai/frozen/hand_board_weights checkpoints/frozen_layers/HandRankClassificationFive --no-sign-request