## cp的时候排除某些文件或者文件夹
rsync -av --exclude=parser/exp parser backup
---
## 创建软连接（只保留一份文件）
ln -s src_path tgt_path
---
## 查找
* find /tmp -atime +21 -and -size +10G
* find . -type f -name '*.php' | xargs grep "include"
* find /var/logs -maxdepth 1 -mtime +7 -ok rm {} \\;

## 设置代理
这里只介绍用科学上网工具clash挂代理
* 在本地电脑查看端口，打开LAN
<img src="imgs/clash_proxy.png" alt="image-20211012160316595" style="zoom:50%;" />

* 在linux设置代理

  export http_proxy=http://本地电脑ip:7890

  export https_proxy=http://本地电脑ip:7890

  export ftp_proxy=http://本地电脑:7890

  (有root权限的可以把上面三行加到**/etc/profile** 末尾，然后**source /etc/profile**；没有权限的就只能在终端设置临时变量了，之后可能是多词设置)

* 注意git remote url要确保是使用的是http协议地址，如：https://github.com/zsLin177/repository.git

## wget
* 后台下载并断点续传: wget -b -c url
  
## zip
zip -vr data.zip data/

## conda 修改镜像源
* 修改~/.condarc为：
<img src="imgs/conda_src.png" alt="image-2" style="zoom:80%;" />

## 创建密钥
* ssh-keygen -t rsa -b 4096 -C "你的邮箱地址"
* 保存路径不要和之前的id_rsa覆盖

## 杀死僵尸进程
因为使用PyTorch设置多线程进行数据读取，其实是假的多线程，他是开了N个子进程（PID都连着）进行模拟多线程工作，所以你的程序跑完或者中途kill掉主进程的话，子进程的GPU显存并不会被释放，需要手动一个一个kill才行，具体方法描述如下：
```shell
fuser -v /dev/nvidia*
# 使用下属命令一句话杀死所有进程
fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
# 如果只想关闭掉某张显卡上的驻留进程，如0号nvidia显卡，那么命令为:
fuser -v /dev/nvidia0 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
# 关闭掉1号显卡上所有的驻留进程：
fuser -v /dev/nvidia1 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
```

  
