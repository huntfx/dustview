@echo off
rem Builds build\XAPOFX1_5.dll (32-bit) with MSVC Build Tools 2022.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars32.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /O2 /MT /W3 /EHa /LD dusthook.cpp d3dlog.cpp /Fe:build\XAPOFX1_5.dll /Fo:build\ /link /DEF:exports.def user32.lib
