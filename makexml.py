import inspect
import os
from xml.dom.minidom import Document
import cv2


def create_xml_element(doc, tag, text=None):
    element = doc.createElement(tag)
    if text is not None:
        text_node = doc.createTextNode(text)
        element.appendChild(text_node)
    return element


def parse_yolo_line(line, image_shape):
    data = line.strip().split(' ')
    class_id = data[0]
    center_x, center_y, width, height = map(float, data[1:])
    x_min = max(0, int((center_x - width / 2) * image_shape[1]))
    y_min = max(0, int((center_y - height / 2) * image_shape[0]))
    x_max = min(image_shape[1], int((center_x + width / 2) * image_shape[1]))
    y_max = min(image_shape[0], int((center_y + height / 2) * image_shape[0]))
    return class_id, x_min, y_min, x_max, y_max


def find_image_path(pic_path, txt_file_name):
    base_name = txt_file_name[:-4]
    for ext in [".jpg", ".bmp", ".png", ".jpeg", ".webp"]:
        img_path = os.path.join(pic_path, base_name + ext)
        if os.path.exists(img_path):
            return img_path, ext
    return None, None


def make_xml(pic_path, txt_path, xml_path, class_dict):
    for path in [pic_path, txt_path, xml_path]:
        os.makedirs(path, exist_ok=True)

    for txt_file_name in os.listdir(txt_path):
        xml_builder = Document()
        annotation = xml_builder.createElement("annotation")
        xml_builder.appendChild(annotation)

        txt_file = open(os.path.join(txt_path, txt_file_name))
        txt_lines = txt_file.readlines()

        img_path, ext = find_image_path(pic_path, txt_file_name)
        if img_path is None:
            print(f"Image not found for {txt_file_name}")
            continue

        image = cv2.imread(img_path)
        image_shape = image.shape

        annotation.appendChild(create_xml_element(xml_builder, "folder", 'images'))
        annotation.appendChild(create_xml_element(xml_builder, "filename", txt_file_name[:-4] + ext))
        annotation.appendChild(create_xml_element(xml_builder, "path", img_path))

        size = create_xml_element(xml_builder, "size")
        for tag, value in zip(["width", "height", "depth"], image_shape):
            size.appendChild(create_xml_element(xml_builder, tag, str(value)))
        annotation.appendChild(size)

        for line in txt_lines:
            class_id, *coords = parse_yolo_line(line, image_shape)
            obj = create_xml_element(xml_builder, "object")
            obj.appendChild(create_xml_element(xml_builder, "name", class_dict[class_id]))
            obj.appendChild(create_xml_element(xml_builder, "pose", "Unspecified"))
            obj.appendChild(create_xml_element(xml_builder, "truncated", "0"))
            obj.appendChild(create_xml_element(xml_builder, "difficult", "0"))

            bndbox = create_xml_element(xml_builder, "bndbox")
            for tag, coord in zip(["xmin", "ymin", "xmax", "ymax"], coords):
                bndbox.appendChild(create_xml_element(xml_builder, tag, str(coord)))
            obj.appendChild(bndbox)

            annotation.appendChild(obj)

        with open(os.path.join(xml_path, txt_file_name[:-4] + ".xml"), 'w') as f:
            xml_builder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')


def main():
    # Example usage:
    print('''
    # Replace with your class dictionary
    class_dict = {'0': 'car', '1': 'person'}  
    pic_path = './pictures/'
    txt_path = './labels/'
    xml_path = './annotations/'
    make_xml(pic_path, txt_path, xml_path, class_dict)
    ''')
    print("请运行main.py")


if __name__ == '__main__':
    main()
