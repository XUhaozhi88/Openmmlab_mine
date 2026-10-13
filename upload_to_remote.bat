@echo off
echo Uploading openmmlab_upload.tar.gz to remote server...
echo.
echo Please enter the password when prompted: J,a*lO1h2zTvJ4-e
echo.
scp -P 26595 C:\Users\xhz\openmmlab_upload.tar.gz root@125.71.97.90:/root/
echo.
if %errorlevel% equ 0 (
    echo Upload successful!
) else (
    echo Upload failed. Please check the connection and try again.
)
pause
