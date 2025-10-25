from pathlib import Path
from utils.gitignore import GitignoreFilter

root = Path('/Users/john/Desktop/johhn/connectNext/backend')
gitignore = GitignoreFilter(root)

# Test what's being filtered
test_files = [
    'node_modules/mongodb/lib/db.js',
    'node_modules/mongodb/README.md', 
    'config/db.js',
    'node_modules/@types/node/sqlite.d.ts',
]

print("Testing gitignore filter:")
for f in test_files:
    path = root / f
    ignored = gitignore.should_ignore(path)
    in_deps = gitignore._is_in_dependency_dir(path)
    is_doc = gitignore._is_dependency_doc(path)
    
    print(f"\n{f}")
    print(f"  In dependency dir: {in_deps}")
    print(f"  Is doc: {is_doc}")
    print(f"  Should ignore: {ignored}")