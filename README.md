# 1 简介
主要包含4个python脚本：foreign.py，makexml.py，remove.py，main.py，通过调用mian.py实现主要功能
主要功能：将前景和背景进行图像融合，并将前景作为背景标签（目标检测中的被标记物体）
# 2 目录组织
## 2.1 目录结构
### background---------------------背景图片路径，需放置
### foreign------------------------背景图片路径，需放置
### data---------------------------自生成过程文件夹
### dataset------------------------自生成过程文件夹，存放前景背景融合后的图片文件和标记文件
### foreign.py---------------------图像融合脚本
### makexml.py---------------------自生成xml脚本
### remove.py----------------------重命名脚本
### main.py------------------------主程序脚本

