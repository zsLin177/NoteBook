# git

### git push -u origin branch 权限不够
* 首先确保ssh-agent正常工作
	- eval $(ssh-agent -s)
	- Agent pid 94457   （正常结果）
* 将私钥id_rsa添加到ssh代理中
	ssh-add ~/.ssh/id_rsa

### 创建新的分支
* 创建并切换到新分支，新分支的代码就是当前分支的代码
git checkout -b span-based

### 列出所有远程分支
git branch -r

### 拉取远程分支并在本地创建新分支
git checkout -b dev origin/dev

### 删除本地分支
git branch -d test

### 拉取远程分支更新本地分支
（需要确保本地分支处于远程分支log中的某一个）
git pull origin main

### 回退到某个版本
git reset --hard 版本号
