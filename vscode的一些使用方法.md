# vscode的一些使用问题
## windows vscode免密连接服务器
* vscode安装Remote-ssh
* Create your local SSH key pair
  - 在powershell输入 ssh-keygen
  - Authorize your Windows machine to connect
    - $USER_AT_HOST="your-user-name-on-host@hostname"
    - $PUBKEYPATH="$HOME\.ssh\id_rsa.pub"
    - $pubKey=(Get-Content "$PUBKEYPATH" | Out-String); ssh "$USER_AT_HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '${pubKey}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
## 终端可以连接服务器，vscode不能连接且报错“_workbench.downloadResource failed”
* 该问题由更新导致
* 先参考网址：https://blog.csdn.net/cuichenghd/article/details/118763047
