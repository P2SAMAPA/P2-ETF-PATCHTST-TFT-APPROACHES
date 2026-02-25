import darts
import os
import glob

darts_path = os.path.dirname(darts.__file__)
print('darts installed at:', darts_path)
print('Searching for patch files:')
patch_files = glob.glob(os.path.join(darts_path, '**', '*patch*'), recursive=True)
if patch_files:
    for f in patch_files:
        print(' -', f)
else:
    print('No patch files found.')
