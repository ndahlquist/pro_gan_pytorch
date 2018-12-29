import os

import av
import cv2
import numpy as np
from tqdm import tqdm
import glob

from parse import parse

files = [s for s in glob.glob("samples/*.png") if os.path.isfile(s)]
files.sort(key=lambda s: os.path.getmtime(s))

im = cv2.imread(files[0])

output_filename = 'out.mp4'
if os.path.exists(output_filename):
    os.remove(output_filename)

output_file = av.open(output_filename, 'w')
output_video_stream = output_file.add_stream("mpeg4", 24)
output_video_stream.width = im.shape[0]
output_video_stream.height = im.shape[1]
output_video_stream.bit_rate = 16000000
output_video_stream.bit_rate_tolerance = 16000000

skip_rate = 1

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, im.shape[1] - 10)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

print("Encoding video.")
for i, filename in enumerate(tqdm(files)):

    if i % skip_rate != 0:
        continue

    im = cv2.imread(filename)
    #im = cv2.resize(im, (512, 512))

    (epoch,) = parse("samples/{:d}.png", filename)

    text = '%d' % (epoch)
    cv2.putText(im, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    cv2.waitKey(10)

    frame = av.VideoFrame.from_ndarray(im, format='bgr24')
    packet = output_video_stream.encode(frame)
    output_file.mux(packet)
output_file.close()