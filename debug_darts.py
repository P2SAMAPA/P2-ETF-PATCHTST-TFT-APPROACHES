import darts
import inspect
import os
import sys

print('darts version:', darts.__version__)
print('darts path:', darts.__file__)
print('Python executable:', sys.executable)

# List files in forecasting directory
forecasting_path = os.path.join(os.path.dirname(darts.__file__), 'models', 'forecasting')
print('\nContents of darts/models/forecasting:')
if os.path.exists(forecasting_path):
    for f in sorted(os.listdir(forecasting_path)):
        if f.endswith('.py') and f != '__init__.py':
            print(' -', f)
else:
    print('Path not found:', forecasting_path)

# Show what's in __init__.py of forecasting
print('\nContents of darts/models/forecasting/__init__.py:')
init_path = os.path.join(forecasting_path, '__init__.py')
if os.path.exists(init_path):
    with open(init_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:20]:  # first 20 lines
            print('   ', line.rstrip())
else:
    print('__init__.py not found.')

# Now try to list all classes available in darts.models.forecasting
print('\nClasses in darts.models.foreaching (via __all__):')
try:
    from darts.models import forecasting
    if hasattr(forecasting, '__all__'):
        print('__all__ =', forecasting.__all__)
    else:
        print('No __all__ attribute.')
    # Also try dir()
    all_members = [name for name in dir(forecasting) if not name.startswith('_')]
    print('Non-private members:', all_members)
except Exception as e:
    print('Could not inspect forecasting module:', e)

# Search for any class with "patch" in the name
print('\nSearching for any class containing "patch":')
try:
    from darts.models import forecasting
    patch_classes = []
    for name, obj in inspect.getmembers(forecasting):
        if inspect.isclass(obj) and 'patch' in name.lower():
            patch_classes.append(name)
    if patch_classes:
        for cls in patch_classes:
            print(' -', cls)
    else:
        print('No class with "patch" found.')
except Exception as e:
    print('Error searching:', e)

# Try to import directly from forecasting module
print('\nTrying from darts.models.forecasting import * :')
try:
    from darts.models.forecasting import *
    print('Imported successfully.')
    # List what's now in locals()
    locals_list = [k for k in locals().keys() if not k.startswith('_')]
    print('Now in namespace:', [k for k in locals_list if 'patch' in k.lower()])
except Exception as e:
    print('Failed:', e)

# Alternative imports
print('\nTrying alternative imports:')
alternative_paths = [
    'darts.models.forecasting.patchtst_model',
    'darts.models.forecasting.patchtst',
    'darts.models.forecasting.patch_tst',
    'darts.models.forecasting.patch_tst_model',
]

for mod_path in alternative_paths:
    try:
        mod = __import__(mod_path, fromlist=['PatchTSTModel'])
        if hasattr(mod, 'PatchTSTModel'):
            print(f'✅ Found PatchTSTModel in {mod_path}')
        else:
            print(f'❌ Module {mod_path} exists but no PatchTSTModel')
    except ImportError as e:
        print(f'❌ Cannot import {mod_path}: {e}')
