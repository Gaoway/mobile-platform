## termux

### 1. 安装
官网`https://github.com/termux/termux-app`

将termux 程序设为后台运行，以便长时间运行：

长按termux图标，选择“耗电管理”，选择“允许完全后台行为”

### 2. ssh配置
```bash
apt upgrade && apt update
apt install openssl-tool
pkg install openssh
sshd
```

查看用户名和IP地址：
```bash
whoami
ifconfig
```

passwd修改密码：
```bash
passwd
```

### 3. ssh连接
```bash
ssh name@IP -p 8022
```

### 4. 安装必要环境
换科大源
```bash
sed -i 's@termux.org/packages/@mirrors.ustc.edu.cn/termux/apt/termux-main@'   $PREFIX/etc/apt/sources.list
pkg update
```
安装python3.8
```bash
pkg install python
pkg install python-pip
pkg install wget proot # proot是一个在termux中运行linux的工具
```