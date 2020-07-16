# Copyright 2020 Petuum. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample utility functions."""

import numpy as np


def uniform_sample_by_choices(choices):
    """
    Uniformly sample an option from a list of options.

    Args:
        choices (list): a list of values to be sampled from.

    Returns:
        choice: the sampled value.

    """
    assert choices
    p = np.random.uniform()
    t = 1.0 / len(choices)
    sample = choices[0]
    for i, c in enumerate(choices):
        if p < t * (i+1):
            sample = c
            break
    return sample


def binary_sample(boundary=0.5):
    p = np.random.uniform()
    if p < boundary:
        return True
    else:
        return False


def sample_merge_group(num_group, num_candidates):

    def is_valid(assignment):
        unique_assignment = np.unique(assignment)
        if unique_assignment.shape[0] == num_group:
            return True
        return False

    assignment = np.random.randint(1, num_group+1, [num_candidates])
    while not is_valid(assignment):
        assignment = np.random.randint(1, num_group+1, [num_candidates])
    return assignment
