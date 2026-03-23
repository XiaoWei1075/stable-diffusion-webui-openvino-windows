
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--skip-torch-cuda-test --precision full --no-half
set PYTORCH_TRACING_MODE=TORCHFX
set USE_OPENVINO=1

::set WEBUI_LAUNCH_LIVE_OUTPUT=1
::set HTTPS_PROXY=http://127.0.0.1:7980
::set HTTP_PROXY=http://127.0.0.1:7980

call webui.bat

