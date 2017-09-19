import sys
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import glob
from tqdm import tqdm

# mac
FONT = '/Library/Fonts/Times New Roman Bold.ttf'
# linux
FONT = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'

image_folder = sys.argv[1]
frames = glob.glob(image_folder+'/*')
#img = Image.open("sample_in.jpg")
for i in tqdm(frames):
    img = Image.open(i)
    #mg.convert('RGB')
    draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype(FONT, 50)
    #font = ImageFont.load_default()
    # draw.text((x, y),"Sample Text",(r,g,b))ls ..
    name = os.path.splitext(os.path.basename(i))[0]
    epoch, batch = name.split('_')
    draw.text((0, 0),'EPOCH:  {}'.format(epoch),('#CCFFFF'),font=font)
    if int(batch) % 10 == 0:
        # format to enchanches handeling with imove
        img.save('./labeled/{0:0>5}{0:0>3}.jpg'.format(int(epoch),batch))
