"""
Launch interactive console for TinyPythonLLM.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from console.interactive import main

if __name__ == "__main__":
    main()
