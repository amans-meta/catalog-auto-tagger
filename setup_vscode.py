#!/usr/bin/env python3
"""
VS Code Environment Setup Helper

This script helps configure VS Code to recognize the correct Python interpreter
and installed packages, fixing the red file errors.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def setup_vscode_environment():
    """Configure VS Code environment for the project"""
    print("🔧 Setting up VS Code Environment")
    print("=" * 50)

    project_root = Path(__file__).parent
    vscode_dir = project_root / ".vscode"

    # Ensure .vscode directory exists
    vscode_dir.mkdir(exist_ok=True)

    # Get Python executable info
    python_exec = "/usr/local/bin/python3"
    pip_path = "/Users/amans/Library/Python/3.12/bin/pip"
    site_packages = "/Users/amans/Library/Python/3.12/lib/python/site-packages"

    print(f"🐍 Python executable: {python_exec}")
    print(f"📦 Site packages: {site_packages}")

    # Create VS Code settings
    settings = {
        "python.defaultInterpreterPath": python_exec,
        "python.pythonPath": python_exec,
        "python.analysis.extraPaths": ["./src", site_packages],
        "python.analysis.autoSearchPaths": True,
        "python.analysis.autoImportCompletions": True,
        "python.analysis.typeCheckingMode": "basic",
        "python.linting.enabled": False,
        "python.terminal.activateEnvironment": False,
        "files.exclude": {"**/__pycache__": True, "**/*.pyc": True},
        "python.envFile": "${workspaceFolder}/.env",
        "python.analysis.stubPath": site_packages,
    }

    settings_file = vscode_dir / "settings.json"
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=4)

    print(f"✅ Created: {settings_file}")

    # Create launch configuration for debugging
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Run Demo",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/demo_catalog_tagger.py",
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}",
                "env": {"PYTHONPATH": "${workspaceFolder}/src"},
            },
            {
                "name": "Run Test System",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/test_system.py",
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}",
                "env": {"PYTHONPATH": "${workspaceFolder}/src"},
            },
        ],
    }

    launch_file = vscode_dir / "launch.json"
    with open(launch_file, "w") as f:
        json.dump(launch_config, f, indent=4)

    print(f"✅ Created: {launch_file}")

    # Verify packages are installed
    print(f"\n📋 Verifying installed packages...")

    try:
        import requests

        print("✅ requests")
    except ImportError:
        print("❌ requests not found")

    try:
        import pandas

        print("✅ pandas")
    except ImportError:
        print("❌ pandas not found")

    try:
        import nltk

        print("✅ nltk")
    except ImportError:
        print("❌ nltk not found")

    try:
        import pydantic

        print("✅ pydantic")
    except ImportError:
        print("❌ pydantic not found")

    # Test project imports
    print(f"\n🧪 Testing project imports...")

    sys.path.insert(0, str(project_root / "src"))

    try:
        from models.product import ProductInfo

        print("✅ models.product")
    except ImportError as e:
        print(f"❌ models.product: {e}")

    try:
        from utils.config import ConfigManager

        print("✅ utils.config")
    except ImportError as e:
        print(f"❌ utils.config: {e}")

    try:
        from utils.text_processing import TextProcessor

        print("✅ utils.text_processing")
    except ImportError as e:
        print(f"❌ utils.text_processing: {e}")

    print(f"\n🎯 Next Steps:")
    print(f"1. Restart VS Code completely (close and reopen)")
    print(f"2. Open Command Palette (Cmd+Shift+P)")
    print(f"3. Run 'Python: Select Interpreter'")
    print(f"4. Choose: {python_exec}")
    print(f"5. Run 'Python: Refresh IntelliSense'")
    print(f"6. Run 'Developer: Reload Window'")

    print(f"\n✅ VS Code setup complete!")

    return True


if __name__ == "__main__":
    try:
        success = setup_vscode_environment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
