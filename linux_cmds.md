# cp的时候排除某些文件或者文件夹
rsync -av --exclude=parser/exp parser backup
---
# 创建软连接（只保留一份文件）
ln -s src_path tgt_path
