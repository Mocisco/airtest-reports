@echo off
chcp 65001 >nul
echo ========================================
echo       画质测试系统 - 防火墙设置
echo ========================================
echo.

echo 正在开放防火墙端口5000...
echo.

:: 检查是否已存在规则
netsh advfirewall firewall show rule name="画质测试系统" >nul 2>&1
if errorlevel 1 (
    echo 添加防火墙规则...
    netsh advfirewall firewall add rule name="画质测试系统" dir=in action=allow protocol=TCP localport=5000
    echo ✅ 防火墙端口5000已开放
) else (
    echo ✅ 防火墙规则已存在
)

echo.
echo 🌐 网络访问地址: http://192.168.94.13:5000
echo.
echo 现在其他人可以通过上面的地址访问你的画质测试系统
echo.
pause