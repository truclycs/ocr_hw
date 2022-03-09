import json
import cv2
import argparse

from pathlib import Path
from predict import Predictor
from PIL import Image, ImageDraw, ImageFont
from utils import load_yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vgg_seq2seq.yml')
    args = parser.parse_args()

    config = load_yaml(args.config)

    detector = Predictor(config=config)
    patterns = ['*.json']
    input_dir = Path('/home/trucly/Downloads/examples')
    paths = []
    for pattern in patterns:
        paths += list(input_dir.glob(f'**/{pattern}'))
    cnt = 0
    for path in paths:
        filename = str(path)
        with open(filename, 'r') as f:
            obj = json.load(f)

        image_name = obj['imagePath']
        image = cv2.imread(str(input_dir.joinpath(image_name)))
        shapes = obj['shapes']
        image = cv2.cvtColor((image), cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil_size = image_pil.size
        white_image = Image.new('RGB', (image_pil_size[0], image_pil_size[1]), (255, 255, 255))

        d = ImageDraw.Draw(white_image)
        for x in shapes:
            cnt += 1
            point = x['points']
            line = image[int(point[0][1]):int(point[1][1]), int(point[0][0]):int(point[1][0])]

            if 'value' in x:
                text = x['value']
            else:
                text, char_probs = detector(line)
            print(text)

            font = ImageFont.truetype("ArialUnicodeMS.ttf", 35)
            d.text((point[0][0]-20, point[0][1]-20), text, fill=(0, 0, 0), font=font)
            # d.line(coor, fill="red", width=4)
            # for point in coor:
            #     d.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="red")

            # d = ImageDraw.Draw(image_pil)
            # d.line(coor, fill="red", width=4)
            # for point in coor:
                # d.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="red")

        merge_image = Image.new('RGB', (2 * image_pil_size[0], image_pil_size[1]), (250, 250, 250))
        merge_image.paste(image_pil, (0, 0))
        merge_image.paste(white_image, (image_pil_size[0], 0))
        # merge_image.save("/home/trucly/Documents/DATASET/example/RESULTS/" + str(image_name))
        merge_image.save("/home/trucly/Downloads/results/" + str(image_name))
