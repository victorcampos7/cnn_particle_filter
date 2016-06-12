import sys
import caffe
import numpy as np

# Global variables
net = None
transformer = None


def load_cnn(deploy_path, caffemodel_path, mean_file_path='ilsvrc_2012_mean.npy'):
    global net, transformer
    net = caffe.Net(deploy_path, caffemodel_path, caffe.TEST)

    # Configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean_file_path).mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)


def get_feature_map(im, layer='conv5'):
    # Make a forward pass
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]), blobs=[layer])
    return out[layer]


def get_feature_map_batch(im_list, layer='conv5', batch_size=64):
    # Pre-process images
    print 'Pre-processing images...'
    sys.stdout.flush()
    preprocessed_images = np.asarray(preprocess_images(im_list))
    np.save('/imatge/vcampos/work/pr_im.npy', preprocessed_images[0])

    num_images = len(im_list)
    end_pointer = 0
    end = False
    features = list()
    # Loop through all the images
    while not end:
        # Update pointers
        start_pointer = end_pointer
        end_pointer = start_pointer + batch_size
        # Check if we are out of range
        if end_pointer >= num_images:
            end_pointer = num_images
            end = True
        # Get batch of images and reshape net
        print 'Start pointer: ', start_pointer
        print 'End pointer: ', end_pointer
        print 'Total images: ', num_images
        sys.stdout.flush()
        batch = preprocessed_images[start_pointer:end_pointer]
        if net.blobs['data'].shape[0] != len(batch):
            change_batch_size(len(batch))
            print 'Successfully changed batch size to %d' % len(batch)
            sys.stdout.flush()
        # Forward pass
        out = net.forward_all(data=batch, blobs=[layer])
        print 'Forward pass done'
        sys.stdout.flush()
        # Store all feature maps in a list
        for feat_map in out[layer]:
            features.append(feat_map)

    return features


def change_batch_size(batch_size):
    data_shape = net.blobs['data'].shape
    net.blobs['data'].reshape(batch_size, data_shape[1], data_shape[2], data_shape[3])
    net.blobs['prob'].reshape(batch_size, )
    net.reshape()  # optional -- the net will reshape automatically before a call to forward()


def preprocess_images(im_list):
    num_images = len(im_list)
    ret_list = [None] * num_images
    for i in range(0, num_images):
        ret_list[i] = transformer.preprocess('data', im_list[i])
    return ret_list
