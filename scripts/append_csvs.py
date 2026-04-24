# Utility script for merging newly generated CSV files into the main dataset CSV file

"""
Append multiple CSV files into the main CSV, skipping each source file's header row.
Creates a timestamped backup of the target before modifying it.

Usage:
    python append_csvs.py

Files are expected to be in the same directory as this script. Adjust the `TARGET` and
`SOURCES` lists below if needed.
"""
from pathlib import Path
import shutil
import datetime

# Edit if filenames differ
TARGET = Path('synthetic_conversations_full.csv')
SOURCES = [
    Path('synthetic_conversations (1).csv'),
    Path('synthetic_conversations(18).csv'),
    Path('synthetic_conversations(20).csv'),
]

if not TARGET.exists():
    print(f"Target file not found: {TARGET.resolve()}")
    raise SystemExit(1)

# Backup target
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
backup = TARGET.with_suffix(TARGET.suffix + f'.bak.{ts}')
shutil.copy2(TARGET, backup)
print(f"Backed up {TARGET} -> {backup}")

# Append each source, skipping its first line (header)
total_appended = 0
with TARGET.open('a', encoding='utf-8', newline='') as out_f:
    for src in SOURCES:
        if not src.exists():
            print(f"Source not found, skipping: {src}")
            continue
        appended = 0
        with src.open('r', encoding='utf-8-sig') as in_f:
            # Skip header
            header = in_f.readline()
            for line in in_f:
                out_f.write(line)
                appended += 1
        total_appended += appended
        print(f"Appended {appended} rows from {src}")

print(f"Done. Total rows appended: {total_appended}")
print(f"Updated file: {TARGET}")
