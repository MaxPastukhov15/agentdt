import os
import sys
from pathlib import Path
from unittest.mock import patch

from app.utils.icon_path import get_resource


def test_get_resource_dev_mode():
    with patch.object(sys, "frozen", False, create=True):
        with patch.object(sys, "_MEIPASS", "", create=True):
            result = get_resource("assets/icon.ico")
            assert result.endswith("assets/icon.ico")


def test_get_resource_frozen_mode():
    fake_meipass = str(Path("/fake/path/_MEIPASS"))
    with patch.object(sys, "frozen", True, create=True):
        with patch.object(sys, "_MEIPASS", fake_meipass, create=True):
            result = get_resource("assets/icon.ico")
            assert os.path.normpath(result) == os.path.normpath(
                fake_meipass + "/assets/icon.ico"
            )
