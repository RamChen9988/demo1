@echo off
echo ========================================
echo   对抗性图像生成项目 - 一键安装脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 显示Python版本
echo 检测到Python版本:
python --version
echo.

REM 检查pip是否安装
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到pip，请确保Python安装完整
    pause
    exit /b 1
)

echo 开始安装项目依赖...
echo.

REM 创建虚拟环境（可选）
set /p create_venv="是否创建虚拟环境？(y/n): "
if /i "%create_venv%"=="y" (
    echo 正在创建虚拟环境...
    python -m venv adversarial-env
    echo 虚拟环境创建完成！
    echo.
    echo 请手动激活虚拟环境:
    echo    adversarial-env\Scripts\activate
    echo 然后再次运行此脚本或执行: pip install -r requirements.txt
    pause
    exit /b 0
)

REM 安装依赖
echo 正在安装依赖包...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   依赖安装成功！
    echo ========================================
    echo.
    echo 现在可以运行以下脚本:
    echo   - adversarial_image_generator.py
    echo   - large_size_adversarial_demo.py  
    echo   - test_image_recognition.py
    echo.
    echo 详细使用方法请参考 INSTALL.md
) else (
    echo.
    echo ========================================
    echo   依赖安装失败！
    echo ========================================
    echo.
    echo 请检查:
    echo   1. 网络连接
    echo   2. Python版本兼容性
    echo   3. 系统权限
)

echo.
pause
