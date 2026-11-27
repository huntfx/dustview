@echo off
rem Builds build\simstep.exe, build\modstep.exe, build\brushstep.exe and build\rcptable.exe (32-bit) with MSVC Build Tools 2022.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars32.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /O2 /MT /W3 /EHa simstep.cpp /Fe:build\simstep.exe /Fo:build\ || exit /b 1
cl /nologo /O2 /MT /W3 /EHa modstep.cpp /Fe:build\modstep.exe /Fo:build\ || exit /b 1
cl /nologo /O2 /MT /W3 /EHa brushstep.cpp /Fe:build\brushstep.exe /Fo:build\ || exit /b 1
cl /nologo /O2 /MT /W3 rcptable.cpp /Fe:build\rcptable.exe /Fo:build\
