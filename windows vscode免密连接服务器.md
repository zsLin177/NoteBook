# windows vscode免密连接服务器

* vscode安装Remote-ssh
* Create your local SSH key pair
  - 在powershell输入 ssh-keygen
  - Authorize your Windows machine to connect
    - $USER_AT_HOST="your-user-name-on-host@hostname"
    - $PUBKEYPATH="$HOME\.ssh\id_rsa.pub"
    - $pubKey=(Get-Content "$PUBKEYPATH" | Out-String); ssh "$USER_AT_HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '${pubKey}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"