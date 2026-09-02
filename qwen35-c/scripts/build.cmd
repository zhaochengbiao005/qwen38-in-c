@echo off
rem build.cmd - configure + build + test (clang-cl / Ninja)
if not exist "%~dp0devcmd.cmd" (echo missing devcmd.cmd & exit /b 1)
call "%~dp0devcmd.cmd"
set "BUILD_DIR=%~dp0..\build"
if not exist "%BUILD_DIR%" (
  cmake -S "%~dp0.." -B "%BUILD_DIR%" -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release
  if errorlevel 1 exit /b 1
)
cmake --build "%BUILD_DIR%" || exit /b 1
rem deploy libomp.dll next to executables (needed at runtime for OpenMP)
if exist "%Q35_LLVM%\bin\libomp.dll" copy /y "%Q35_LLVM%\bin\libomp.dll" "%BUILD_DIR%\" >nul 2>&1
ctest --test-dir "%BUILD_DIR%" --output-on-failure
exit /b %errorlevel%
