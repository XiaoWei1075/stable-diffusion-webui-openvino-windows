# Copyright (C) 2024-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import launch
import os
from pathlib import Path
repo_root = Path(__file__).parent
req = repo_root / "requirements.txt"
# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")


def install():
    # install_requirements()
    launch.run_pip(
        f'install -r "{req}"',
        "requirements for OpenVINO",
        live=default_command_live,
    )

    print("OpenVINO extension install complete")


install()
