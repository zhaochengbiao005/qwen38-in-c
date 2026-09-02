@echo off
rem devcmd.cmd - enter dev environment: VS Build Tools (VC/SDK) + LLVM clang-cl
if not defined Q35_VS_BT set "Q35_VS_BT=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
if not defined Q35_LLVM  set "Q35_LLVM=C:\Program Files\LLVM"
call "%Q35_VS_BT%\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b 1
set "PATH=%Q35_LLVM%\bin;%PATH%"
set "PATH=C:\Program Files\CMake\bin;C:\Users\123456\AppData\Local\Microsoft\WinGet\Links;%PATH%"

