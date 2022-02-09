import os
import shutil
import sys
import re

in_dir = sys.argv[1]

for f in os.listdir(in_dir):
    if f == 'get.sh' or re.match('.*\-log.*', f):
        continue

    cat,phrase,gender,num = f.split('_')
    if phrase == 'male' or phrase == 'female':
        continue

    new_f = f'{cat}_{gender}_{phrase}_{num}'

    shutil.move(os.path.join(in_dir, f),
                os.path.join(in_dir, new_f))
