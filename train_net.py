#!/usr/bin/python
# -*- encoding: utf-8 -*-


import argparse
from core.train import train


def main():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
        '--cfg', dest='cfg', required=True,
        help='configuration file')
    args = parser.parse_args()

    train(args.cfg)

if __name__ == '__main__':
    main()


