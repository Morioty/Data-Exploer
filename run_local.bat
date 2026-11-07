@echo off
rem Local-only launch (not reachable from other machines)
"%~dp0.venv\Scripts\streamlit.exe" run "%~dp0app.py" --server.address localhost
