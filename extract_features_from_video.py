import os
import sys
import numpy as np
import argparse
from helpers import img_helper, caffe_helper

# Constants
img_size = 227

# Parse arguments
parser = argparse.ArgumentParser(description='Extract convolutional features for a video')
parser.add_argument('-p', '--path', help='folder containing the frames (a subfolder will be created with the features)')
parser.add_argument('-d', '--deploy', help='path to the deploy.prototxt file')
parser.add_argument('-c', '--caffemodel', help='path to the caffemodel file')
parser.add_argument('-m', '--mean_file', help='path to the mean file')

args = parser.parse_args()
video_path = args.path
mean_file_path = args.mean_file
deploy_path = args.deploy
caffemodel_path = args.caffemodel
if not video_path.endswith('/'):
    video_path += '/'
mean_values = np.load(mean_file_path).mean(1).mean(1)


# List images in the given path
print 'Listing images...'
sys.stdout.flush()
filenames = list()
img_extensions = ["png", "bmp", "jpg", "jpeg"]
for f in os.listdir(video_path):
    file_extension = f.split('.')[-1]
    if file_extension.lower() in img_extensions:
        filenames.append(video_path+f)
print "Listed %d images" % len(filenames)


# Load images
print 'Loading images...'
sys.stdout.flush()
num_images = len(filenames)
images = [None] * num_images
for i in range(0, num_images):
    images[i] = (img_helper.resize_image(img_helper.load_image(filenames[i]), img_size, mean_values=mean_values))
print 'Loaded %d images' % len(images)
sys.stdout.flush()


# Load CNN
print 'Loading CNN...'
sys.stdout.flush()
caffe_helper.load_cnn(deploy_path, caffemodel_path, mean_file_path=mean_file_path)


# Extract features
print 'Extracting features...'
sys.stdout.flush()
feature_maps = caffe_helper.get_feature_map_batch(images, layer='conv5', batch_size=256)


# Save feature maps in disk
save_dir = video_path + 'features/'
print 'Saving features to disk...'
print 'Images: %d' % num_images
print 'Feature maps: %d' % len(feature_maps)
sys.stdout.flush()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print 'Created ', save_dir
    sys.stdout.flush()
for i in range(0, num_images):
    filename = filenames[i].split('/')[-1].split('.')[0]
    save_file = save_dir + filename + '.npy'
    np.save(save_file, feature_maps[i])

print 'Done!'
sys.stdout.flush()
