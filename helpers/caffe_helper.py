import caffe

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
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    return out[layer]


def get_feature_map_batch(im_list, layer='conv5', batch_size=64):
    num_images = len(im_list)
    end_pointer = -1
    end = False
    features = list()
    # Loop through all the images
    while not end:
        # Update pointers
        start_pointer = end_pointer + 1
        end_pointer = start_pointer + batch_size
        # Check if we are out of range
        if end_pointer >= num_images:
            end_pointer = num_images - 1
            end = True
        # Get batch of images and reshape net
        batch = im_list[start_pointer:end_pointer]
        if net.blobs['data'].shape[0] != len(batch):
            change_batch_size(len(batch))
        # Forward pass
        out = net.forward_all(data=np.asarray([transformer.preprocess('data', batch)]))
        # Store all feature maps in a list
        for feat_map in out[layer]:
            features.append(feat_map)

    return features


def change_batch_size(batch_size):
    data_shape = net.blobs['data'].shape
    net.blobs['data'].reshape(batch_size, data_shape[1], data_shape[2], data_shape[3])
    net.blobs['label'].reshape(batch_size, )
    net.reshape()  # optional -- the net will reshape automatically before a call to forward()
