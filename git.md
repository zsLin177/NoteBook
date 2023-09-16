# git

### git push -u origin branch 权限不够
* 首先确保ssh-agent正常工作
	- eval $(ssh-agent -s)
	- Agent pid 94457   （正常结果）
* 将私钥id_rsa添加到ssh代理中
	ssh-add ~/.ssh/id_rsa

### 创建新的分支
* 创建并切换到新分支，新分支的代码就是当前分支的代码，且新分支的历史和当前分支一致
  git checkout -b span-based
* 创建新分支，新分支的代码就是当前分支的代码，但是历史为空
  git checkout --orphan new_branch_name

### push本地新分支到远程仓库
git push origin mtl-span-info:mtl-span-info

### 列出所有远程分支
git branch -r

### 拉取远程分支并在本地创建新分支
* git checkout -b dev origin/dev
* git fetch origin remote-branch-name:local-branch-name


### 删除本地分支
git branch -d test 或者强制删除：-D

### 删除远程分支
git push origin --delete test

### 拉取远程分支更新本地分支
（需要确保本地分支处于远程分支log中的某一个）
git pull origin main

### git重新命名本地分支
* 如果对于分支不是当前分支，可以使用下面代码：
  git branch -m "原分支名" "新分支名"

* 如果是当前，那么：

  git branch -m "新分支名称"

### 回退到某个版本
git reset --hard 版本号

### 查看版本号
git log 

### 重新设置远程仓库地址
git remote set-url origin new_address

### 添加远程仓库地址
git remote add origin xxx.git

### 查看用户配置
* git config --list --global
* git config --list --local

### 设置用户名和邮箱
* git config --local user.name "name"
* git config --local user.name "xxx@xxx"
* git config --global user.name "name"
