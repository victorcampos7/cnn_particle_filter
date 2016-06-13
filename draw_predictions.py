import os
import sys
import argparse
from helpers import img_helper
import re

# Parse arguments
parser = argparse.ArgumentParser(description='Track object using pre-computed feature maps')
parser.add_argument('-p', '--path', help='folder containing the frames (a subfolder will be created for the tracking)')

args = parser.parse_args()
video_path = args.path
if not video_path.endswith('/'):
    video_path += '/'
predictions_file_path = video_path + 'predictions.txt'


# Check that the tracking has been done
if not os.path.isfile(predictions_file_path):
    sys.exit('Error: predictions not found')


# List images in the given path
print 'Listing images...'
sys.stdout.flush()
non_sorted_filenames = list()
img_extensions = ["png", "bmp", "jpg", "jpeg"]
for f in os.listdir(video_path):
    file_extension = f.split('.')[-1]
    if file_extension.lower() in img_extensions:
        non_sorted_filenames.append(f)
print "Listed %d images" % len(non_sorted_filenames)


digits = re.compile(r'(\d+)')


def tokenize(filename):
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))


filenames = sorted(non_sorted_filenames, key=tokenize)
#filenames = sorted(non_sorted_filenames, key=lambda n: int(n.split('.')[0]))

# Open ground truth file
try:
    gt_file = open(video_path+'gt.txt', 'r')
except:
    sys.exit("Error: could not open ground truth file in %s" % video_path+'gt.txt')

# Get bounding boxes from the ground truth
gt_bounding_boxes = list()
while True:
    line = gt_file.readline()
    if len(line) == 0:
        break
    values = line.split(',')
    # Parse coordinates (x,y,w,h) as integers
    x0, y0, x1, y1 = int(float(values[0])), int(float(values[1])), int(float(values[2])), int(float(values[3]))
    gt_bounding_boxes.append((x0, y0, x1 - x0, y1 - y0))


# Get bounding boxes from the predictions
predictions_file = open(predictions_file_path, 'r')
bounding_boxes = list()
while True:
    line = predictions_file.readline()
    if len(line) == 0:
        break
    values = line.split(',')
    # Parse coordinates (x,y,w,h) as integers
    x, y, w, h = int(values[0]), int(values[1]), int(values[2]), int(values[3])
    bounding_boxes.append((x, y, w, h))
predictions_file.close()


# Create folder for the new images
save_dir = video_path + 'predictions/'
if not os.path.exists(save_dir):
    print 'Creating directory: %s' % save_dir
    os.makedirs(save_dir)


# Draw bounding boxes
for i, filename in enumerate(filenames):
    if i % 10 == 0:
        print 'Creating image %d of %d' % (i+1, len(filenames))
        sys.stdout.flush()
    img_helper.draw_bounding_box(video_path=video_path,
                                 filename=filename,
                                 save_dir=save_dir,
                                 pred_bounding_box=bounding_boxes[i],
                                 gt_bounding_box=gt_bounding_boxes[i])
