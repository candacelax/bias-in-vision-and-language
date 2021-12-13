#!/usr/bin/env python
import os
import re

if __name__ == '__main__':
    for test_dir in os.listdir():
        if not os.path.isdir(test_dir):
            continue

        attr_dirs = [re.match(d) for d in ]
