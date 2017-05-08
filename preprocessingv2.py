from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Rewritten by: Omar Mousa
# Referenced from Alex's Preprocessing file
# Googles Bulid_an_image
# and the referenced tensormodel script all combined

import os
import time
import random
import sys
import threading
import itertools
import numpy
import tensorflow as tf
#import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from scipy.misc import imread
from PIL import Image
from multiprocessing import Pool

"""
Resizing images from dirctory for preprocessing
"""

def imageResize(args):
    image = Image.open(args[0] + "/" + args[1])
    dim = args[2].split('x')
    image = image.resize((int(dim[0]), int(dim[1])), Image.ANTIALIAS)
    image.save('./resized_Images/' + args[2] + '/' + args[1])

def PCA(source, num_components, chuck_size):
    image_path = sorted(list(source), key = lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    size, images = 0, []
    n_chunks = len(image_path)//chunk_size
    pca = IncrementalPCA(n_components=num_components, batch_size=chunk_size)
    for i in range(n_chunks):
        print('Chunk:', i, '\tIndex:', i * chunk_size + size)
        while size < chunk_size:
            images.append(imread(source+image_path[i * chunk_size + size]).flatte())
            size += 1
        pca.partial_fit(np.asarray(images))
        size, images = 0, []

        if i == n_chunks - 1:
            i += 1
            print('chunk:', i, 'index:', i * chunk_size + size)
            transformed = pca.transform(np.asarray(images))
            if xTransformed is None:
                xTransformed = transformed
            else:
                xTransformed = np.vstack((xTransformed, transformed))
            size, images = 0, []
            if i == n_chunks - 1:
                i += 1
                while i * chunk_size + size < len(image_path):
                    images.append(imread(source+image_path[i * chunk_size]).flatten())
                    size += 1
                trasformed = pca.transform(np.asarray(images))
                xTransformed = np.vstack((xTransformed, transformed))
            print("\nTransformed matrix shape:", xTransformed.shape)
            return xTransformed
        if __name__ == "__main__":
            source = './train/right'
            new_size = '32x32'
            pool = Pool()
            start = time.time()
            pool.map(imageResize, zip(itertools.repeat(source), listdir(source), itertools.repeat(new_size)))
            print("Resized Images in {0} seconds".formate(time.time() - start))
"""
Building a tensor record from the image batches
"""
tf.app.flags.DEFINE_string('train_directory', './train',
                           'Training data directory')
tf.app.flags.DEFINE_string('validating_directory', './train',
                           'Validating data directory')
#tf.app.flags.DEFINE_string('test_directory', './test',)
max_shard_bytes  = 64000000
tf.app.flags.DEFINE_integer('numberOfshards', max_shard_bytes - 50 ,
                            'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_integer('numberOfthreads', 10000,
                            'Number of threads to process the images')


tf.app.flags.DEFINE_string('labels_file', ' ', 'labels file')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    """inserts int 64 features into proto"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """inters bytes of features into proto"""
    return tf.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_trRecord(fileName, buffer4Image, label, text, height, width):
    """Builds proto"""

    color = 'RGB'
    channel = 3
    image_formate = 'JPEG'

    image = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/color': _bytes_feature(tf.compat.as_bytes(color)),
        'image/channel': _int64_feature(channel),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(os.basename(fileName))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(buffer4Image))}))
    return image

class codedImage(object):
    """ Helper class that provides to help TensorFlow """

def __init___(self):
    self._sess = tf.Session()

    self._decode_jpeg_data = tf.placeholder(dtype = tf.string)
    self.decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data,channel=3)

def decode_jpeg(self, dataForimage):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: dataForimage})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def is_jpeg(fileName):
    return '.jpeg' in fileName


def processTheimage(fileName, imagecode):
    with tf.gfile.FastGFile(fileName, 'rb') as f:
        dataForimage = f.read()

    image = color.decode_jpeg(dataForimage)

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

def batch_image_files(imagecode, threadIndex, _range, name, fileName,
                       text, labels, numberOfshards):
    numberOfthreads = len(_range)
    assert not numberOfshards % numberOfthreads
    numberOfshardsPerBatch = int(numberOfshards / numberOfthreads)

    shardRange = np.linspace(_range[threadIndex][0],
                             _range[threadIndex][1],
                             numberOfshardsPerBatch + 1).astype(int)
    numberOffilesInThread = _range[threadIndex][1] - _range[threadIndex][0]

    count = 0
    for s in range(numberOfshardsPerBatch):
        _shardSize = threadIndex * numberOfshardsPerBatch + x
        fileNameOut = '%s-%.5d - of - %.5d' % (name, _shard, numberOfshards)
        outFile = os.path.join(FLAGS.output_directory, output_filename)
        write = tf.python_io.TFRecordWrite(outFile)

        shardCount = 0
        fileInShard = np.arrange(shardSize[x], shardSize[x + 1], dtype = int)
        for i in fileInShard:
            _fileName = fileName[i]
            _label = labels[i]
            _text = text[i]

        try:
            bufferImage, height, width = processTheimage(fileName, imagecode)
        except Exception as err:
            print(err)
            print('Unexpected decoding error so %s was skipped:' % fileName)
            continue

        image = _convert_to_trRecord(fileName, buffer4Image, label,
                                     text, height, width)
        shardCount += 1
        count += 1

        if not count % 1000:
            print('%s [thread %d]: Processed %d of %d images in this batch of threads' % (time.time(), threadIndex, count, numberOffilesInThread))
            sys.stdout.flush()
        writer.closer()

        print('%s [thread %d]: Wrote %d images to %s' % (time.time(), threadIndex, shardCount, outFile))
        sys.stdout.flush()

        shardCount = 0
    print('%s [thread %d]: Wrote %d images to %d shards' % (time.time(), threadIndex, count, numberOffilesInThread))
    sys.stdout.flush()
def processImages(title, namesOffiles, txt, label_, numberOfshards):

    assert len(namesOffiles) == len(txt)
    assert len(namesOffiles) == len(label_)

    _spacing= np.linspace(0, len(namesOffiles), FLAGS.num_threads + 1).astype(np.int)
    range_ = []
    for i in range(len(_spacing) - 1):
        range_.append([spacing[i], spacig[i + 1]])
    print('Launching %d thread to space: %s' % (FLAGS.num_threads, range_))
    sys.stdout.flush()

    coord = tf.train.Coordinator()
    encoder = codedImage()

    threaded = []
    for threadIndex in range(len(range_)):
        args = (encoder, threadIndex, range_, title, namesOffiles, txt, label_, numberOfshards)
        thred = threading.Thread(target = batch_image_files, args = args)
        thred.start()
        threaded.append(threds)

    coord.join(threaded)
    print('%d Images writting in Data Set' % (time.time(), len(namesOffiles)))
    sys.stdout.flush()

def findImages(dir_, labeledFile):
    print('Acquiring Input labels and images from %s' % dir)
    uniqueLabels = [y.strip() for y in tf.gfile.FastGFile(labeledFile, 'r').readlines()]

    labeled = []
    nameOffiles = []
    txt = []

    labeledIndex = 1
    for text in uniqueLabels:
        jpegPath = '%s/%s/*' % (dir_, txt)
        fileMatch = tf.gfile.Glob(jpegPath)
        labeled.extend([labelIndex] * len(fileMatch))
        txt.extend([txt] * len(fileMatch))
        nameOffiles.extend(fileMatch)

    if not labelIndex % 100:
        print('Found files in %d of %d classes' % (labelIndex, len(labeled)))
        labelIndex += 1

    shuffleIndecies = list(range(len(nameOffiles)))
    random.seed(1234567890)
    random.shuffle(shuffleIndecies)

    namesOffiles = [namesOffiles[i] for i in shuffleIndecies]
    txt = [txt[i] for i in shuffleIndecies]
    labes = [labes[i] for i in shuffleIndecies]

    print('%d JPEG files found across %d with labels inside %s' % (len(nameOffiles), len(unique), dir_))
    return nameOffiles, txt, labes

def dataset(title, directory, numberOfshards, labesFile):
    nameOffiles, txt, labes, = findImages(directory,labels_file)
    processImages(title, nameOffiles, txt, lebes, numberOfshards)

def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, ('Make FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, ('Make FLAGS.num_threads commensurate with FLAGS.validation_shards')
    print('Storing into %s' % FLAGS.output_directory)

    dataset(('validation', FLAGS.validation_directory), FLAGS.validation_shards, FLAGS.labels_file)
    dataset(('train', FLAGS.train_directory), FLAGS.train_shards, FLAGS.labels_file)

    if __name__ == '__main__':
        tf.app.run()


'''
The model
'''

#nClass = 2

height = 224
width = 224

def get_Image(nameOFfile):
    filename_queue = tf.train.string_input_producer([nameOFfile], num_epochs = 10)

    reader = tf.TFRecordReader()

    key, image = reader.read(nameOFfile)

    features = tf.parse_single_example(image, features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([],tf.int64),
        'image/colorspace': tf.FixedLenFeature([],dtype=tf.string, default_value = ''),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value = ''),
        'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value = ''),
        'image/format': tf.FixedLenFeature([],dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

    labe = features['image/class/label']
    _buffer = features['image/encoded']

    with tf.name_scope('decode_jpeg', [_buffer], None):
        pic = tf.image.decode_jpeg(_buffer, channels=3)

        pic = tf.image.convert_image_dtype(pic, dtype=tf.float32)

        pic = tf.reshape(1-tf.image.rgb_to_grayscale(image), [height*width])

        labe = tf.stack(tf.one_hot(labe-1, nClass))

        return labe, pic
labe, pic = get_Image("data/train-00000-of-00001")

validLabe ,validImage = get_Image("data/validation-00000-of-00001")

batchI, batchL = tf.train.shuffle_batch([pic,labe], batch_size=1000, capacity=2000, min_after_dequeue = 1000)

batchVI, batchVL = tf.train.shuffle_batch([batchVI, batchVL], batch_size = 1000, capacity = 2000, min_after_dequeue = 1000)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, width*height])

y_ = tf.placeholder(tf.float32, [None, nClass])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# set up "vanilla" versions of convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#print "Running Convolutional Neural Network Model"
nFeatures1=32
nFeatures2=64
nNeuronsfc=1024

# use functions to init weights and biases
# nFeatures1 features for each patch of size 5x5
# SAME weights used for all patches
# 1 input channel
W_conv1 = weight_variable([5, 5, 1, nFeatures1])
b_conv1 = bias_variable([nFeatures1])

# reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth
# means 1 colour channel per pixel
# x_image = tf.reshape(x, [-1,28,28,1])
x_image = tf.reshape(x, [-1,width,height,1])


# hidden layer 1
# pool(convolution(Wx)+b)
# pool reduces each dim by factor of 2.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# similarly for second layer, with nFeatures2 features per 5x5 patch
# input is nFeatures1 (number of features output from previous layer)
W_conv2 = weight_variable([5, 5, nFeatures1, nFeatures2])
b_conv2 = bias_variable([nFeatures2])


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# denseley connected layer. Similar to above, but operating
# on entire image (rather than patch) which has been reduced by a factor of 4
# in each dimension
# so use large number of neurons

# check our dimensions are a multiple of 4
#if (width%4 or height%4):
  #print "Error: width and height must be a multiple of 4"
  #sys.exit(1)

W_fc1 = weight_variable([(width/4) * (height/4) * nFeatures2, nNeuronsfc])
b_fc1 = bias_variable([nNeuronsfc])

# flatten output from previous layer
h_pool2_flat = tf.reshape(h_pool2, [-1, (width/4) * (height/4) * nFeatures2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# reduce overfitting by applying dropout
# each neuron is kept with probability keep_prob
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# create readout layer which outputs to nClass categories
W_fc2 = weight_variable([nNeuronsfc, nClass])
b_fc2 = bias_variable([nClass])

# define output calc (for each class) y = softmax(Wx+b)
# softmax gives probability distribution across all classes
# this is not run until later
y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# measure of error of our model
# this needs to be minimised by adjusting W and b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# start training
nSteps=5000
for i in range(nSteps):

  batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

  # run the training step with feed of images
  if simpleModel:
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
  else:
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


  if (i+1)%100 == 0: # then perform validation

    # get a validation batch
    vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
    if simpleModel:
      train_accuracy = accuracy.eval(feed_dict={
        x:vbatch_xs, y_: vbatch_ys})
    else:
      train_accuracy = accuracy.eval(feed_dict={
        x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i+1, train_accuracy))


# finalise
coord.request_stop()
coord.join(threads)
