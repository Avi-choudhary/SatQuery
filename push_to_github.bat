@echo off
title SatQuery - Push to GitHub
cd /d "%~dp0"
echo ================================================================
echo   SatQuery AI - GitHub Publisher
echo   Target: https://github.com/Avi-choudhary/SatQuery
echo ================================================================
echo.
echo Pushing commits to GitHub master branch...
echo (If prompted, please sign in with your GitHub account)
echo.
git push -u origin master
echo.
if %ERRORLEVEL% EQU 0 (
    echo ================================================================
    echo   SUCCESS! SatQuery AI has been successfully pushed to GitHub!
    echo ================================================================
) else (
    echo ================================================================
    echo   Push encountered an error. Please check your credentials.
    echo ================================================================
)
echo.
pause
