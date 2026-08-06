import sys

from .cli import main

# Windows consoles default to cp1252, which mangles quotes and accents in
# fetched sources.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

raise SystemExit(main())
