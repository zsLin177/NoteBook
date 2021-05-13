# git

### git push -u origin branch 权限不够
* 首先确保ssh-agent正常工作
	- eval $(ssh-agent -s)
	- Agent pid 94457   （正常结果）
* 将私钥id_rsa添加到ssh代理中
	ssh-add ~/.ssh/id_rsa