import os
import sys
import numpy as np
import argparse
from helpers import img_helper, particle_filter
import re

# Parse arguments
parser = argparse.ArgumentParser(description='Track object using pre-computed feature maps')
parser.add_argument('-p', '--path', help='folder containing the frames (a subfolder will be created for the tracking)')
parser.add_argument('-n', '--num_particles', help='number of particles for the tracker')

args = parser.parse_args()
video_path = args.path
num_particles = int(args.num_particles)
if not video_path.endswith('/'):
    video_path += '/'
feat_dir = video_path + 'features/'


# List to store the predicted bounding boxes
predictions = list()


# Check that the features have been extracted
if not os.path.exists(feat_dir):
    sys.exit('Error: feature maps not found')


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
    gt_bounding_boxes.append((x0, y0, x1-x0, y1-y0))


# Reference for the tracker
initial_bb = gt_bounding_boxes[0]
x0, y0, w0, h0 = initial_bb[0], initial_bb[1], initial_bb[2], initial_bb[3]
predictions.append('%d,%d,%d,%d' % (x0, y0, w0, h0))

# Create particle filter
img_dims = np.shape(img_helper.load_image(video_path+filenames[0]))
initial_feature_map = np.load(feat_dir + filenames[0].split('.')[0] + '.npy')
tracker = particle_filter.ParticleFilter(num_particles, x0, y0, w0, h0, 20, img_dims)
tracker.set_model(initial_feature_map, x0, y0, w0, h0)


# Track
num_frames = len(gt_bounding_boxes)
print '\n\n------------------ Frame %d/%d ------------------' % (1, num_frames)
print 'Ground truth: %d, %d, %d, %d' % (x0, y0, w0, h0)
print 'Prediction:   %d, %d, %d, %d' % (x0, y0, w0, h0)
for i in range(1, num_frames):
    print '\n\n------------------ Frame %d/%d ------------------' % (i + 1, num_frames)
    current_feature_map = np.load(feat_dir + filenames[i].split('.')[0] + '.npy')
    x, y, w, h = tracker.track(current_feature_map, estimation='max')
    predictions.append('%d,%d,%d,%d' % (x, y, w, h))
    gt_bb = gt_bounding_boxes[i]
    x_gt, y_gt, w_gt, h_gt = gt_bb[0], gt_bb[1], gt_bb[2], gt_bb[3]
    print 'Ground truth: %d, %d, %d, %d' % (x_gt, y_gt, w_gt, h_gt)
    print 'Prediction: %d, %d, %d, %d' % (x, y, w, h)
    sys.stdout.flush()


# Write predictions to a file
save_file = video_path+'predictions.txt'
print 'Saving predictions to %s' % save_file
sys.stdout.flush()
output_file = open(save_file, 'w')
for bb in predictions:
    output_file.write(bb+'\n')
output_file.close()
