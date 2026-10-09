import os
import tempfile

# main.py opens the persistent index when imported; point it at a throwaway directory
# so tests never read or write the real data/index.
os.environ["INDEX_DIR"] = tempfile.mkdtemp(prefix="evidencerag-test-index-")
