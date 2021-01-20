import os
import json
import numpy as np
import shutil
import argparse


def convert(size: list, box: list) -> list:
    """
    将标注的 xml 文件生成的【左上角x,左上角y,右下角x，右下角y】标注转换为yolov5训练的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def handle_json(file_path):
    info_list = []
    with open(file_path) as f:
        infos = json.load(f)
        for info in infos:
            info_list.append(info)
    print("OK")
    return info_list






def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f'Created folder: {dir}')

def gen_label():
    pass


def handle_data(file_path, list_json):
    """

    :param list_json:
    :param file_path: 数据集的根目录
    :return: None
    """
    list_dirs = os.listdir(file_path)
    len_dirs = len(list_dirs)
    for list_dir, step in zip(list_dirs, range(0, len_dirs)):
        if step <= len_dirs * 0.8:
            temp_image_path = pre_image_train_data_path
            temp_label_path = pre_label_train_data_path
        else:
            temp_image_path = pre_image_val_data_path
            temp_label_path = pre_label_val_data_path
        for list_json_item in list_json:
            if list_dir == list_json_item['name']:
                size = [list_json_item['image_height'], list_json_item['image_width']]
                category = list_json_item['category']
                box = list_json_item['bbox']
                [x, y, w, h] = convert(size, box)
                fstr = f"{category} {x} {w} {y} {h}\n"
                label_path = os.path.join(temp_label_path, str(step)) + '.txt'
                with open(label_path, 'a') as f:
                    f.writelines(fstr)
        image_path = os.path.join(temp_image_path, str(step)) + '.jpg'
        raw_image_path = os.path.join(file_path, list_dir)
        print(f"Generating file:{image_path}")
        shutil.copyfile(raw_image_path, image_path)



def create_dirs(dirs):
    """
    :param file_path: 字符串列表
    :return:
    """
    for dir in dirs:
        create_dir(dir)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--json', type=str, default='./tile_round1_train_20201231/train_annos.json', help='initial weights path')
    # parser.add_argument('--')
    # opt = parser.parse_args()

    project_path = "./tile_round1_train_20201231"
    json_path = os.path.join(project_path, "train_annos.json")
    raw_data_path = os.path.join(project_path, "train_imgs")
    pre_data_path = os.path.join(project_path, "pre_data")

    pre_image_data_path = os.path.join(pre_data_path, "image")
    pre_label_data_path = os.path.join(pre_data_path, "label")
    pre_image_train_data_path = os.path.join(pre_image_data_path, "train")
    pre_image_val_data_path = os.path.join(pre_image_data_path, "val")
    pre_label_train_data_path = os.path.join(pre_label_data_path, "train")
    pre_label_val_data_path = os.path.join(pre_label_data_path, "val")

    list_paths = [pre_data_path, pre_label_data_path, pre_image_data_path, pre_image_train_data_path,
                  pre_image_val_data_path, pre_label_train_data_path, pre_label_val_data_path]

    create_dirs(list_paths)

    list_json = handle_json(json_path)

    handle_data(file_path=raw_data_path, list_json=list_json)




