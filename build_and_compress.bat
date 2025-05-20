@echo off
set IMAGE_NAME=qwen-runner
set IMAGE_TAG=8b
set TAR_NAME=%IMAGE_NAME%-%IMAGE_TAG%.tar
set GZ_NAME=%TAR_NAME%.gz

echo building Docker image %IMAGE_NAME%:%IMAGE_TAG% ...
docker build -t %IMAGE_NAME%:%IMAGE_TAG% .

if errorlevel 1 (
    echo  building image failed, please check dockerfile or internet
    pause
    exit /b 1
)

echo built successfully, export as %TAR_NAME% ...
docker save %IMAGE_NAME%:%IMAGE_TAG% -o %TAR_NAME%

if not exist %TAR_NAME% (
    echo  export failed %TAR_NAME%
    pause
    exit /b 1
)

echo compressing as %GZ_NAME% ...
gzip -f %TAR_NAME%

if exist %GZ_NAME% (
    echo had compressed %GZ_NAME%
) else (
    echo compress failed, please check gzip is equipt
)

pause

