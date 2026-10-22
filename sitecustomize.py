"""Desktop-only subprocess defaults: never allocate extra console windows.

``start_gui.py`` launches the API with ``pythonw.exe``. Any console-subsystem
child (Flower's generated ``flwr-*`` launchers, for example) would otherwise get
a new black window. ``sitecustomize`` is imported by every Python process whose
``PYTHONPATH`` contains the application root, so one hook covers the whole tree.
"""

import os
import subprocess


if os.name == "nt" and os.environ.get("FL_BACKDOOR_DESKTOP") == "1":
    _CREATE_NO_WINDOW = 0x08000000
    _CREATE_NEW_CONSOLE = 0x00000010
    _BasePopen = subprocess.Popen

    class _NoWindowPopen(_BasePopen):
        def __init__(self, *args, **kwargs):
            flags = kwargs.get("creationflags") or 0
            if not flags & _CREATE_NEW_CONSOLE:
                kwargs["creationflags"] = flags | _CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = _NoWindowPopen
