import os
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
filenames = list()
img_extensions = ["png", "bmp", "jpg", "jpeg"]
for f in os.listdir(video_path):
    file_extension = f.split('.')[-1]
    if file_extension.lower() in img_extensions:
        img_extensions.append(video_path+f)


# Load and preprocess images
num_images = len(filenames)
images = [None] * num_images
for i in range(0, num_images):
    images.append(img_helper.resize_image(img_helper.load_image(filenames[i]), img_size, mean_values=mean_values))


# Load CNN
caffe_helper.load_cnn(deploy_path, caffemodel_path, mean_file_path=mean_file_path)


# Extract features
feature_maps = caffe_helper.get_feature_map_batch(images)


# Save feature maps in disk
save_dir = video_path + 'features/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print 'Created ', save_dir
for i in range(0, num_images):
    save_file = save_dir + filenames[i].split('.')[0] + '.npy'
    np.save(save_file, feature_maps[i])
