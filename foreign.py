import inspect
import cv2
import numpy as np
import os  # 导入处理操作系统相关功能的模块

# 全局默认配置信息
image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".raw", ".psd"]  # 支持的图片格式列表
annotation_formats = [".xml", ".json", "txt"]


class PaseForeign:
    def __init__(self,
                 background_path,
                 foreign_path,
                 pic_path,
                 txt_path,
                 foreign_choice_num,
                 foreign_alpha_range,
                 error_path,
                 img_size=512,
                 empty=False):
        self.img_size = img_size
        self.pic_path = pic_path
        self.txt_path = txt_path
        # 产品图
        self.path_image_bg = background_path  # 样品图片
        self.list_image_bg = os.listdir(self.path_image_bg)
        # 异物图
        self.path_image_foreign = foreign_path  # 异物图片
        self.list_image_foreign = os.listdir(self.path_image_foreign)

        # 图像融合参数
        self.num_foreign = len(self.list_image_foreign)
        self.foreign_choice_num = foreign_choice_num
        self.alpha_range = foreign_alpha_range
        self.error_path = error_path
        self.empty = empty

    # 旋转图像
    @staticmethod
    def rotate(image, border_value):
        # 获取图像的高度和宽度
        h, w = image.shape[:2]
        # 生成一个随机旋转角度
        angle = np.random.uniform(low=0, high=360)
        # 构建旋转矩阵
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        cos = np.abs(mat[0, 0])
        sin = np.abs(mat[0, 1])
        # 计算旋转后图像的新宽度和高度
        w_new = h * sin + w * cos
        h_new = h * cos + w * sin
        mat[0, 2] += (w_new - w) * 0.5
        mat[1, 2] += (h_new - h) * 0.5
        w_new = int(np.round(w_new))
        h_new = int(np.round(h_new))
        # 进行仿射变换，实现图像旋转
        image = cv2.warpAffine(src=image, M=mat, dsize=(w_new, h_new), borderValue=border_value)
        return image

    # 裁剪背景
    def crop_background(self, image):
        # 获取图像的高度和宽度
        h, w = image.shape[:2]
        h = max(h, 512)
        w = max(w, 512)
        # 随机生成裁剪后的高度和宽度
        h_random = np.random.randint(self.img_size, h + 1)
        w_random = np.random.randint(self.img_size, w + 1)

        # 随机生成裁剪的起始点坐标
        y_random = np.random.randint(h - h_random + 1)
        x_random = np.random.randint(w - w_random + 1)
        # 对图像进行裁剪
        image = image[y_random:y_random + h_random, x_random:x_random + w_random]
        return image

    # 裁剪异物
    @staticmethod
    def crop_foreign(image):
        # 找到图像中像素值大于等于127的位置索引
        indices = np.where(image >= 35)  # 获取灰度大于指定值的部分，进而划分最小矩形块
        # print(indices)
        # 获取最小和最大的行坐标
        h1 = indices[0].min()
        h2 = indices[0].max()
        # 获取最小和最大的列坐标
        w1 = indices[1].min()
        w2 = indices[1].max()
        # 对图像进行裁剪
        image = image[h1:h2 + 1, w1:w2 + 1]
        return image

    # 放置异物产生标签，次要操作
    def paste(self, image_background, image_foreign, label_n):
        h1, w1 = image_background.shape[:2]
        h2, w2 = image_foreign.shape[:2]
        # 灰度值低于平均值减去20的位置 # 拆分x和y的索引，# 获取x和y坐标范围 （无穷鸡腿图像，鸡骨头位置）
        indices = np.where(image_background < image_background.mean() - 20)
        x_indices, y_indices = indices
        try:
            min_x, max_x = x_indices.min(), x_indices.max()
            min_y, max_y = y_indices.min(), y_indices.max()
        except:
            return image_background, None
        # 全图范围
        # x_random = np.random.randint(w1 - w2 + 1)
        # y_random = np.random.randint(h1 - h2 + 1)
        # 1 异物放置范围
        try:
            y_random = np.random.randint(max(0, min_y - 20), min(max_y + 20, h1 - h2 + 1))
            x_random = np.random.randint(max(0, min_x - 20), min(max_x + 20, w1 - w2 + 1))
        except:
            # print("min_x:", min_x, "max_x:", max_x, "min_y:", min_y, "max_y:", max_y)
            # print(image_foreign.shape)
            # print(max(0, min_y - 20), min(max_y + 20, h1 - h2 + 1))
            # print(max(0, min_x - 20), min(max_x + 20, w1 - w2 + 1))
            return image_background, None

        # 2 判定是否放置
        image_background_roi = image_background[y_random:y_random + h2, x_random:x_random + w2]  # 提取背景图像上对应位置的ROI区域
        if self.empty:
            if image_background_roi.mean() >= 245:  # 如果原图ROI区域平均值大于等于245，则返回原背景图像和空标签
                return image_background, None
        image_foreign = 255 - image_foreign  # 将异物图像进行反色处理
        if image_foreign.astype('float32').min() + 5 > image_background_roi.astype('float32').mean():
            return image_background, None

        # # 获取 image_foreign 和 image_background_roi 的形状
        # shape_foreign = image_foreign.shape
        # shape_background_roi = image_background_roi.shape
        # # 确定在每个维度上的最小形状
        # final_shape = tuple(max(s1, s2) for s1, s2 in zip(shape_foreign, shape_background_roi))
        #
        # # 使用广播进行逐元素比较
        # mask = (image_foreign.astype('float32')[:final_shape[0], :final_shape[1]] < image_background_roi.astype(
        #     'float32'))

        # 3 放置异物(图像融合)
        mask = (image_foreign.astype('float32') < image_background_roi.astype('float32'))  # 生成蒙版，用于合成图像
        # mask = (image_foreign < image_background_roi)  # 生成蒙版，用于合成图像

        image_foreign = image_foreign * mask + image_background_roi * (1 - mask)  # 根据蒙版合成图像
        alpha = np.random.uniform(low=self.alpha_range[0], high=self.alpha_range[1])  # 生成随机的alpha值
        image_background_roi = image_foreign * (1 - alpha) + image_background_roi * alpha  # 混合背景图像和合成图像

        image_background[y_random:y_random + h2, x_random:x_random + w2] = image_background_roi  # 将合成图像放回背景图像上

        # 4 计算异物放置位置和类别标签，并写入
        xc = x_random + w2 / 2 - 0.5  # 计算合成标签的x中心坐标
        yc = y_random + h2 / 2 - 0.5  # 计算合成标签的y中心坐标
        # label_dict = {'points': 0, 'shortlines': 1, 'balls': 2, 'blocks': 3, 'longlines': 4, 'rings': 5,
        #               'longstrips': 6, 'screws': 7, 'nuts': 8, 'clips': 9, 'springs': 10,'foreign': 0}
        label_dict = {'foreign': 0, 'block': 1, 'longline': 2, 'ring': 3, 'screw': 4, 'shape': 5}
        label_index = label_dict.get(label_n, -1)  # 根据传入的标签名获取对应的标签索引，若找不到则返回-1
        size_w = max(0.015, w2 / w1)  # 确保尺寸比值不小于0.015
        size_h = max(0.015, h2 / h1)  # 确保尺寸比值不小于0.015
        if label_index == -1:  # 如果找不到对应的标签索引
            print("未知标签：", label_n)
            label_index = 0  # 默认将标签索引设为0
        label = [label_index, (xc / w1), (yc / h1), size_w, size_h]  # 构建标签信息
        return image_background, label

    # 将异物放到背景上，主要操作
    def process_image(self, image_background):
        flag_bg = 1
        flag_foreign = 1
        id_foreign = 0
        # 1 对背景图像进行裁剪和旋转
        if flag_bg:
            if np.random.randint(2):  # 随机对背景图像进行水平翻转
                image_background = image_background[:, ::-1]
            if np.random.randint(2):  # 对背景图像进行旋转
                image_background = self.rotate(image_background, 255)  # 对背景图像进行旋转
            image_background = self.crop_background(image_background)  # 裁剪背景图像
            image_background = cv2.resize(image_background, [self.img_size, self.img_size], cv2.INTER_AREA)  # 调整背景图像大小
            # image_background = letterbox(image_background, (self.img_size, self.img_size))
        labels = []  # 存储标签信息的列表
        h_b, w_b = image_background.shape[:2]
        # 2 在一张图片上添加多个异物
        # 2.1对异物图像做图像变换
        for _ in range(self.foreign_choice_num):  # 控制异物数量,6,15,20,10
            id_foreign = np.random.randint(self.num_foreign)  # 随机选择一个异物
            image_foreign_path_file = os.path.join(self.path_image_foreign, self.list_image_foreign[id_foreign])
            # image_foreign_path_file = self.path_image_foreign + self.list_image_foreign[id_foreign]
            if os.path.isfile(image_foreign_path_file):
                # image_foreign = cv2.imread(self.path_image_foreign + self.list_image_foreign[id_foreign],
                #                            cv2.IMREAD_GRAYSCALE)
                # 解决中文路径乱码问题
                image_foreign = cv2.imdecode(
                    np.fromfile(file=image_foreign_path_file, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE)
                # 获取异物的类型
                label_n = image_foreign_path_file.rsplit('\\', 1)[-1].split('_')[0]
                h, w = image_foreign.shape[:2]
            else:
                continue
            if h * w < 10 and min(h, w) < 4:
                continue  # 异物图片太小了，不做处理
            # 2.2对异物进行处理
            if flag_foreign:
                if np.random.randint(2):  # 随机对异物图像进行膨胀操作
                    image_foreign = cv2.dilate(image_foreign, np.ones(np.random.randint(1, 3, 2), np.uint8),
                                               iterations=1)
                if np.random.randint(2):  # 随机对异物图像进行水平翻转
                    image_foreign = image_foreign[:, ::-1]
                if np.random.randint(2):  # 随机对异物图像进行垂直翻转
                    image_foreign = image_foreign[::-1, :]
                if np.random.randint(2):  # 旋转90度
                    image_foreign = cv2.rotate(image_foreign, cv2.ROTATE_90_CLOCKWISE)
                image_foreign = self.rotate(image_foreign, 0)  # 对异物图像进行旋转
                image_foreign = self.crop_foreign(image_foreign)  # 裁剪异物图像
            # 2.3 异物大小调整（根据产品）
            if h_b < 513 or w_b < 513:
                h = int(image_foreign.shape[0] * 50 / 100)
                w = int(image_foreign.shape[1] * 50 / 100)
            if label_n == "foreign":
                h, w = h - 1, w - 1
                if h > 20 and w > 20:
                    h, w = h - 10, w - 10
                if h > 30 and w > 30:
                    h, w = h - 20, w - 20
            image_foreign = cv2.resize(image_foreign, (max(w, 1), max(h, 1)),
                                       interpolation=cv2.INTER_AREA)  # 调整异物图像大小
            # 3 将异物粘贴到背景图像上
            image_background, label = self.paste(image_background, image_foreign, label_n)  # 将异物粘贴到背景图像上并生成标签
            # 4 将标签信息添加到列表中
            if label is not None:  # 将标签信息添加到列表中
                labels.append(label)
        # except:
        #     with open(self.error_path + r"\\" + 'error.txt', 'a') as f:  # 记录错误异物图片信息
        #         f.write(self.list_image_foreign[id_foreign] + '\n')
        #     continue
        # 5 返回处理后的图像和标签
        image_background = cv2.cvtColor(image_background, cv2.COLOR_GRAY2BGR)  # 将背景图像转换为RGB格式
        return image_background, np.array(labels).reshape([-1, 5])  # 返回处理后的图像和标签

    def parse_main(self):
        # 1 选择单张背景
        i = np.random.randint(len(self.list_image_bg))
        # 解决中文乱码问题 24.7.11
        image0 = cv2.imdecode(
            np.fromfile(file=os.path.join(self.path_image_bg, self.list_image_bg[i]), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE)
        # 2 将异物粘贴到背景图像上，并获取处理后的图像和标签
        image, labels = self.process_image(image0.copy())
        # 3 构建图像路径和标签路径
        image_path = os.path.join(self.pic_path, self.list_image_bg[i])
        label_path = os.path.join(self.txt_path, os.path.splitext(self.list_image_bg[i])[0] + ".txt")
        # 4 保存图像到指定路径
        cv2.imencode('.jpg', image)[1].tofile(image_path)
        # 5 将标签写入到文件中
        with open(label_path, 'w') as f:
            for label in labels:
                # 将标签信息写入文件
                f.write(f"{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n")


def print_method_signatures(cls):
    # 打印类的方法签名,以下是PaseForeign类的方法

    print(f"Class: {cls.__name__}")
    for name, member in inspect.getmembers(cls):
        if inspect.isfunction(member) or inspect.ismethod(member):
            # Check if the member belongs to the class or its base classes
            if member.__qualname__.split('.')[0] == cls.__name__:
                signature = inspect.signature(member)
                print(f"{name}: {signature}")


if __name__ == '__main__':
    # 打印该类包含的函数名称，以及传递的参数名称
    print_method_signatures(PaseForeign)
    print("请运行main.py")
