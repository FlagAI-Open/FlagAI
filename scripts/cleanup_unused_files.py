#!/usr/bin/env python3
"""
Cleanup Unused Files Script
Identifies and optionally removes unused files from the FlagAI project

This script identifies:
1. macOS resource fork files (._*)
2. Python cache files (__pycache__)
3. Potentially unused files
4. Duplicate/backup files
"""
import os
import sys
import glob
from pathlib import Path
from typing import List, Set


def find_macos_resource_forks(root_dir: str) -> List[str]:
    """Find macOS resource fork files (._*)."""
    resource_forks = []
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        if '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.startswith('._'):
                full_path = os.path.join(root, file)
                resource_forks.append(full_path)
    return resource_forks


def find_pycache_dirs(root_dir: str) -> List[str]:
    """Find __pycache__ directories."""
    pycache_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in root:
            pycache_dirs.append(root)
    return pycache_dirs


def find_potentially_unused_files(root_dir: str) -> List[str]:
    """Find potentially unused files."""
    unused_files = []
    
    # Files that might be unused
    patterns = [
        '*.pyc',
        '*.pyo',
        '*.bak',
        '*.backup',
        '*.swp',
        '*.swo',
        '*~',
    ]
    
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(root_dir, '**', pattern), recursive=True):
            # Skip certain directories
            if '__pycache__' in file_path or '.git' in file_path:
                continue
            unused_files.append(file_path)
    
    return unused_files


def find_duplicate_files(root_dir: str) -> List[str]:
    """Find duplicate/backup files."""
    duplicates = []
    
    # Look for backup files
    backup_patterns = [
        '*_backup.*',
        '*_old.*',
        '*_bak.*',
        '*.orig',
    ]
    
    for pattern in backup_patterns:
        for file_path in glob.glob(os.path.join(root_dir, '**', pattern), recursive=True):
            if '.git' in file_path:
                continue
            duplicates.append(file_path)
    
    return duplicates


def find_upgrade_docs(root_dir: str) -> List[str]:
    """Find upgrade documentation files that might be outdated."""
    upgrade_docs = []
    
    upgrade_patterns = [
        'UPGRADE*.md',
        '*升级*.md',
        'FLASH_ATTN_UPGRADE.md',
    ]
    
    for pattern in upgrade_patterns:
        for file_path in glob.glob(os.path.join(root_dir, pattern)):
            upgrade_docs.append(file_path)
    
    return upgrade_docs


def check_file_usage(file_path: str, root_dir: str) -> bool:
    """Check if a file is imported or used anywhere."""
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    
    # Skip certain files
    if file_name.startswith('._') or file_name.startswith('__'):
        return False
    
    # Check if file is imported
    import_patterns = [
        f'from {file_name_no_ext}',
        f'import {file_name_no_ext}',
        f'{file_name}',
    ]
    
    # Simple check - look for imports in Python files
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                if full_path == file_path:
                    continue
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Check if file is referenced
                        if file_name in content or file_name_no_ext in content:
                            return True
                except:
                    pass
    
    return False


def main():
    """Main function to identify and optionally remove unused files."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("FlagAI File Cleanup Script")
    print("=" * 60)
    print(f"Scanning directory: {root_dir}\n")
    
    # Find different types of files
    print("1. Finding macOS resource fork files (._*)...")
    resource_forks = find_macos_resource_forks(root_dir)
    print(f"   Found {len(resource_forks)} resource fork files")
    
    print("\n2. Finding __pycache__ directories...")
    pycache_dirs = find_pycache_dirs(root_dir)
    print(f"   Found {len(pycache_dirs)} __pycache__ directories")
    
    print("\n3. Finding potentially unused files...")
    unused_files = find_potentially_unused_files(root_dir)
    print(f"   Found {len(unused_files)} potentially unused files")
    
    print("\n4. Finding duplicate/backup files...")
    duplicates = find_duplicate_files(root_dir)
    print(f"   Found {len(duplicates)} duplicate/backup files")
    
    print("\n5. Finding upgrade documentation files...")
    upgrade_docs = find_upgrade_docs(root_dir)
    print(f"   Found {len(upgrade_docs)} upgrade documentation files")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total files to review: {len(resource_forks) + len(pycache_dirs) + len(unused_files) + len(duplicates)}")
    
    # Show some examples
    if resource_forks:
        print(f"\nExample resource fork files (showing first 5):")
        for f in resource_forks[:5]:
            print(f"  - {f}")
    
    if upgrade_docs:
        print(f"\nUpgrade documentation files:")
        for f in upgrade_docs:
            print(f"  - {f}")
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    print("Options:")
    print("1. Delete macOS resource fork files (._*)")
    print("2. Delete __pycache__ directories")
    print("3. Delete potentially unused files")
    print("4. Delete duplicate/backup files")
    print("5. Show all files (dry run)")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        print(f"\nDeleting {len(resource_forks)} resource fork files...")
        for f in resource_forks:
            try:
                os.remove(f)
                print(f"  Deleted: {f}")
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        print("Done!")
    
    elif choice == '2':
        print(f"\nDeleting {len(pycache_dirs)} __pycache__ directories...")
        for d in pycache_dirs:
            try:
                import shutil
                shutil.rmtree(d)
                print(f"  Deleted: {d}")
            except Exception as e:
                print(f"  Error deleting {d}: {e}")
        print("Done!")
    
    elif choice == '3':
        print(f"\nDeleting {len(unused_files)} potentially unused files...")
        for f in unused_files:
            try:
                os.remove(f)
                print(f"  Deleted: {f}")
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        print("Done!")
    
    elif choice == '4':
        print(f"\nDeleting {len(duplicates)} duplicate/backup files...")
        for f in duplicates:
            try:
                os.remove(f)
                print(f"  Deleted: {f}")
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        print("Done!")
    
    elif choice == '5':
        print("\nAll files found:")
        print("\nResource fork files:")
        for f in resource_forks:
            print(f"  - {f}")
        print("\n__pycache__ directories:")
        for d in pycache_dirs:
            print(f"  - {d}")
        print("\nPotentially unused files:")
        for f in unused_files:
            print(f"  - {f}")
        print("\nDuplicate/backup files:")
        for f in duplicates:
            print(f"  - {f}")
    
    else:
        print("Exiting without changes.")


if __name__ == '__main__':
    main()

