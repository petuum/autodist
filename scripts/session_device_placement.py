"""
A script that parses a TensorFlow `log_device_placement` output and prints the device assignments.

To use it, simply save the output of an AutoDist run (with the TF option enabled) to a file and run this script with
that as the argument.
"""
import argparse
import os
import re
from collections import defaultdict


def parse_log_file(log_file, csv_file=None):
    """
    Open and parse a log file to extract the device placements.

    Args:
        log_file (str): Path to the log file
        csv_file (str): Path to save the device placement info as csv format.
    """
    regex = r".+tensorflow/core/common_runtime/placer\.cc:\d+] (.+): (.+)"

    with open(log_file, "r") as f:
        s = f.read()

    matches = re.finditer(regex, s, re.MULTILINE)

    ops = defaultdict(set)

    for matchNum, match in enumerate(matches, start=1):
        ops[match.group(1)].add(match.group(2))

    if csv_file:
        with open(csv_file, 'w+') as f:
            for op, locations in sorted(ops.items()):
                f.write('{},{}\n'.format(op, next(iter(locations))))
        print('Parsed session device placements are saved in: {}'.format(csv_file))
    else:
        for op, locations in sorted(ops.items()):
            print(f"{op}: {locations if len(locations) > 1 else next(iter(locations))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        help="log file path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--csvfile",
        help="csv file path to persist the device placement info",
        type=str,
    )

    FLAGS, unparsed = parser.parse_known_args()

    log_file_path = os.path.abspath(FLAGS.logfile)

    parse_log_file(FLAGS.logfile, FLAGS.csvfile)
