import os
import argparse
import shutil
import sys
from pathlib import Path

from remove import MoveRename
from foreign import PaseForeign
from makexml import make_xml

current_directory = os.path.dirname(os.path.abspath(__file__)) + '\\'
ROOT = os.getcwd() + '\\'
# ROOT = current_directory
print("项目路径为：", ROOT)


def path_init(
        bg=ROOT + "background/",
        fg=ROOT + "foreign/",
        project=ROOT + "data",
        pic=ROOT + "data/images",
        txt=ROOT + "data/label",
        error=ROOT + "data/error",
        xml=ROOT + "data/Annotations",
        target=ROOT + "dataset",
        class_labels=None,
        alpha=(0.2, 0.7),
        number=5,
        fname='',
        fnum='',
        target_rename_flag=False):
    if class_labels is None:
        class_labels = {'0': "foreign", '1': "block", '2': "longline", '3': "ring", '4': "screw", '5': "shape"}

    for_name = fname
    for_num = fnum

    if for_name != '':
        # print("正在处理：for_name", for_name)
        # print("正在处理：for_num", for_num)
        # 清理 for_name 和 for_num 中的路径分隔符
        for_name = for_name.replace('/', '').replace('\\', '')
        for_num = for_num.replace('/', '').replace('\\', '')
        fg = os.path.join(ROOT + "foreign\\", for_name, for_num).rstrip('\\')
        # 去除fg末尾"\"
    # print("正在处理 fg：", fg)

    for dir_i in [pic, txt, xml]:  # 清空删除目录
        if os.path.exists(dir_i):  # 检查目录是否存在
            shutil.rmtree(dir_i)  # 删除目录及其内容
        os.makedirs(dir_i, exist_ok=True)  # 创建目录
    os.makedirs(error, exist_ok=True)
    last_component = os.path.basename(fg)
    # 检查最后一个部分是否以数字结尾
    if last_component[-1].isdigit():
        # 如果最后一个部分是数字结尾，则选择倒数第二个部分和最后一个部分组合
        components = fg.rsplit('\\', 2)
        fix_name_label = components[-2] + last_component
    else:
        # 否则直接使用最后一个部分
        fix_name_label = last_component
    # print("正在处理 fix_name_label：", fix_name_label)
    if target_rename_flag:
        os.makedirs('dataset', exist_ok=True)
        shutil.rmtree('dataset')  # 删除目录及其内容
    os.makedirs('dataset', exist_ok=True)
    loader = PaseForeign(bg, fg, pic, txt, number, alpha, error, img_size=512, empty=False)
    return loader, bg, pic, txt, xml, class_labels, project, target, fix_name_label


def parse_opt(flag=0):
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser(description='This script processes images to extract features.')
    parser.add_argument("--bg", type=str, default=ROOT + "background/can_he/can_he1", help="Background images path")
    parser.add_argument("--fg", type=str, default=ROOT + "foreign", help="Foreign images path")
    parser.add_argument("--project", type=str, default=ROOT + "data", help="Project path")
    parser.add_argument("--pic", "--img", "--images", type=str, default=ROOT + "data/images/", help="Images path")
    parser.add_argument("--txt", "--label", type=str, default=ROOT + "data/label", help="label of txt path")
    parser.add_argument("--xml", type=str, default=ROOT + "data/Annotations", help="label of xml path")
    parser.add_argument("--error", type=str, default=ROOT + "data/error", help="Try exception file path")
    parser.add_argument("--target", type=str, default=ROOT + "data/out", help="Path to target file")
    parser.add_argument("--target_rename_flag", default=True,
                        help="Retain previous target files (including images and annotation files)")
    parser.add_argument("--class_labels", nargs="+", type=str, default=None,
                        help="Class labels (e.g., '0':'foreign', '1':'block')")
    parser.add_argument("--alpha", nargs="+", type=float, default=(0.2, 0.7),
                        help="Adjust transparency for overlaying foreign images")
    parser.add_argument("--number", type=int, default=5, help="Number of foreign images to place")
    parser.add_argument("--fname", type=str, default='blocks', help="Subdirectory folder")
    parser.add_argument("--fnum", type=str, default='1', help="Subdirectory split folder")

    opt = parser.parse_args()
    if flag:
        print("Here is the list of command-line argument information.")
        for arg in vars(opt):
            print(f"--{arg:<13}:{str(getattr(opt, arg)):<60}:")
    return opt


def foreign_all(opt):
    """放置异物函数"""
    """放置异物函数"""
    opt.target_rename_flag = False
    k = 0
    stages1 = [
        ('blocks', [str(j) for j in range(1, 4)], 5, (0.2, 0.7)),
        ('foreigns', [str(j) for j in range(1, 5)], 10, (0.2, 0.7)),
    ]
    stages2 = [
        ('longlines', '', 5, (0, 0.3)),
        ('rings', '', 5, (0.4, 0.6)),
        ('screws', '', 5, (0.2, 0.6)),
        ('shapes', '', 5, (0.4, 0.8)),
    ]
    for i in range(1):
        for f_name, f_files, f_choice_num, f_alpha_range in stages1:  # 含有子文件夹
            for f_file in f_files:
                # print(f_name, f_file, f_choice_num, f_alpha_range)
                opt.fname = str(f_name)
                opt.fnum = str(f_file)
                opt.number = f_choice_num
                opt.alpha = f_alpha_range
                foreign_one(opt)
                k += 1
                print('第{:2}轮文件已完成: 异物图片---{}'.format(k, f_name.lstrip('\\').lstrip('/') + f_file))
        for f_name, f_files, f_choice_num, f_alpha_range in stages2:  # 不含子文件夹
            # 判断当前阶段是否应该执行
            opt.fname = str(f_name)
            opt.fnum = str(f_files)
            opt.number = f_choice_num
            opt.alpha = f_alpha_range
            foreign_one(opt)
            k += 1
            print('第{:2}轮文件已完成: 异物图片---{}'.format(k, f_name.lstrip('\\').lstrip('/') + f_files))


def foreign_one(opt):
    # 定义支持的图片格式和标注格式
    # 1.路径信息初始化
    loader, bg, pic, txt, xml, class_labels, project, target, fix_name = path_init(**vars(opt))
    # 2.实例化类
    for _ in range(len(os.listdir(bg))):
        loader.parse_main()
    # 3.将txt文件转换为xml文件
    make_xml(pic, txt, xml, class_labels)
    # 4.移动文件并重命名
    move = MoveRename(project, target)
    move.move_start(fix_name)


if __name__ == '__main__':
    opt = parse_opt()
    opt.target_rename_flag = True
    opt.fname = ''
    opt.fnum = ''
    opt.bg = 'background/empty_bg'
    opt.fg = 'foreign/screws'
    opt.class_labels = {'0': "foreign", '1': "block", '2': "longline", '3': "ring", '4': "screw", '5': "shape"}
    # 使用反斜杠 \ 分割路径字符串，并获取最后一个部分
    print("背景路径 bg：", opt.bg)
    foreign_all(opt)
