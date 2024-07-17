import numpy as np
import os  # 导入处理操作系统相关功能的模块
import shutil  # 导入用于文件操作的模块
import xml.etree.ElementTree as Et  # 导入用于解析XML文件的模块并使用ET作为别名
import json  # 导入处理JSON数据的模块


class MoveRename:
    def __init__(self, root_path, target_path):
        # 初始化
        self.root_path = root_path  # 源文件路径
        self.target_path = target_path  # 目标文件路径
        # 计算得到的路径
        self.annotation_file = None
        self.image_file = None
        self.target_label_path = None
        self.target_img_path = None
        self.image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".raw", ".psd"]  # 支持的图片格式列表
        self.annotation_formats = [".xml", ".json", "txt"]

    # 检查标注文件是否为空，并移动文件
    def check_and_move(self, annotation_file, image_file):
        """根据标注文件的类型，判断标注文件是否为空，并移动图片文件和标注文件到目标路径"""

        def is_content_non_empty(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                return content.strip()  # 判断内容是否为空（去除空白字符后）

        if annotation_file.endswith('.xml'):  # 如果标注文件是XML格式
            if is_content_non_empty(annotation_file):
                tree = Et.parse(annotation_file)  # 解析XML文件
                root = tree.getroot()  # 获取XML文件的根节点
                objects = root.findall('object')  # 查找所有的 'object' 标签
                if objects:  # 如果存在 'object' 标签
                    shutil.move(annotation_file, self.target_label_path)  # 移动标注文件到目标路径
                    shutil.move(image_file, self.target_img_path)  # 移动图片文件到目标路径
        elif annotation_file.endswith('.json'):  # 如果标注文件是JSON格式
            if is_content_non_empty(annotation_file):
                flag = 0
                with open(annotation_file, 'r') as f:  # 打开标注文件
                    data = json.load(f)  # 加载JSON数据
                    shapes = data.get('shapes', [])  # 提取JSON数据中的'shapes'字段
                    for shape in shapes:  # 遍历'shapes'
                        if 'label' in shape and shape['label']:  # 如果'shape'中包含'label'字段且不为空
                            # print(shape['label'])
                            # print(annotation_file)
                            flag = 1
                            break  # 跳出循环
                if flag:
                    shutil.move(image_file, self.target_img_path, copy_function=shutil.copy2)  # 移动图片文件到目标路径
                    shutil.move(annotation_file, self.target_label_path, copy_function=shutil.copy2)  # 移动标注文件到目标路径
        elif annotation_file.endswith('.txt'):  # 如果标注文件是txt格式
            if is_content_non_empty(annotation_file):
                shutil.move(annotation_file, self.target_label_path)  # 移动标注文件到目标路径
                shutil.move(image_file, self.target_img_path)  # 移动图片文件到目标路径

    # 移动非空图片和标签
    def move_path_check(self):
        """ 源文件路径和目标路径检查 """
        root_path = self.target_path
        target_path = os.path.join(os.getcwd(), "dataset")
        data_folder = os.path.join(root_path, "")  # 判断文件夹位置
        img_path = os.path.join(data_folder, "images") if os.path.exists(data_folder) else root_path
        label_file = os.path.join(data_folder, "Annotations") if os.path.exists(data_folder) else root_path

        target_img_path = os.path.join(target_path, "images")  # 目标图片路径
        target_label_path = os.path.join(target_path, "Annotations")  # 目标标注路径

        os.makedirs(target_img_path, exist_ok=True)  # 创建目标图片路径
        os.makedirs(target_label_path, exist_ok=True)  # 创建目标标注路径
        self.target_label_path = target_label_path
        self.target_img_path = target_img_path
        for root, dirs, files in os.walk(label_file):  # 遍历指定路径下的所有文件和文件夹
            for file in files:  # 遍历文件列表
                # print(file)
                if any(file.endswith(ext) for ext in self.annotation_formats):  # 如果文件是支持的标注格式之一
                    annotation_file = os.path.join(root, file)  # 构造标注文件的完整路径
                    # print(annotation_file)
                    image_file = None  # 初始化图片文件路径为None
                    for ext in self.image_extensions:  # 遍历支持的图片格式列表
                        image_filename = os.path.splitext(os.path.basename(annotation_file))[0] + ext
                        potential_image_file = os.path.join(img_path, image_filename)  # 用标注文件名字拼接为图片名称
                        if os.path.exists(potential_image_file):  # 如果图片文件存在
                            image_file = potential_image_file  # 更新图片文件路径
                            # print(image_file)
                            break  # 跳出循环
                    # # print(image_file)
                    if image_file:  # 如果存在图片文件
                        self.check_and_move(annotation_file, image_file)  # 调用函数检查并移动文件

    # 重命名并移动文件
    def rename_and_move(self, label_name):
        """重命名图片和标签，并移动文件"""
        root_path = self.root_path
        target_path = self.target_path
        # 定义图片路径和标注文件路径
        img_path = os.path.join(root_path, "images")
        label_path = os.path.join(root_path, "Annotations")

        # 定义目标图片路径和目标标注文件路径
        target_img_path = os.path.join(target_path, "images")
        target_json_path = os.path.join(target_path, "Annotations")

        # 确保目标文件夹存在
        os.makedirs(target_img_path, exist_ok=True)
        os.makedirs(target_json_path, exist_ok=True)

        # 遍历图片文件夹中的文件
        for index, img_filename in enumerate(os.listdir(img_path)):
            img_name, img_ext = os.path.splitext(img_filename)

            # 如果是支持的图片文件格式
            if img_ext.lower() in self.image_extensions:
                found_annotation = False

                # 检查是否存在匹配的标注文件
                for annotation_format in self.annotation_formats:
                    annotation_filename = img_name + annotation_format
                    if os.path.exists(os.path.join(label_path, annotation_filename)):
                        found_annotation = True
                        r1 = np.random.randint(1, 9)
                        r2 = np.random.randint(1, 9)
                        r3 = np.random.randint(1, 9)
                        r4 = np.random.randint(1, 9)
                        # 新的文件名格式
                        name_img, ext_img = os.path.splitext(img_filename)
                        name_txt, ext_txt = os.path.splitext(annotation_filename)

                        # 重命名图片文件和标注文件
                        if not img_filename.startswith('_'):  # 添加判断，如果文件名不是以"_"开头
                            new_img_filename = f"{name_img}_{label_name}_{index + 1}_{r1}{r2}{r3}{r4}{ext_img}"
                            new_annotation_filename = f"{name_txt}_{label_name}_{index + 1}_{r1}{r2}{r3}{r4}{ext_txt}"
                            os.rename(os.path.join(img_path, img_filename), os.path.join(img_path, new_img_filename))
                            os.rename(os.path.join(label_path, annotation_filename),
                                      os.path.join(label_path, new_annotation_filename))
                        else:
                            new_img_filename = f"{img_filename}"
                            new_annotation_filename = f"{annotation_filename}"

                        # 移动文件到目标文件夹
                        shutil.move(os.path.join(img_path, new_img_filename),
                                    os.path.join(target_img_path, new_img_filename))
                        shutil.move(os.path.join(label_path, new_annotation_filename),
                                    os.path.join(target_json_path, new_annotation_filename))
                        break
                # 如果没有找到匹配的标注文件
                if not found_annotation:
                    print(f"No matching annotation file found for {img_filename}")

    def move_start(self, label_name):
        """ 只需要标签的类别 """
        self.rename_and_move(label_name)
        self.move_path_check()
if __name__ == '__main__':
    print("请运行main.py")

