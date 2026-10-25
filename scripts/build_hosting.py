"""Copy canonical docs assets to the dist directory supported by Sites hosting."""
from pathlib import Path
import shutil

r = Path(__file__).resolve().parents[1]
shutil.copytree(r / 'docs', r / 'dist', dirs_exist_ok=True)
print('Prepared dist/ from docs/')
