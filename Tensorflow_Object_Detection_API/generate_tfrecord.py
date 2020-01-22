import tensorflow as tf
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(label_and_data_info):
height = None # Image height
width = None # Image width
filename = None # Filename of the image. Empty if image is not from file
encoded_image_data = None # Encoded image bytes
image_format = None # b'jpeg' or b'png'

xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
xmaxs = [] # List of normalized right x coordinates in bounding box
            # (1 per box)
ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
ymaxs = [] # List of normalized bottom y coordinates in bounding box
            # (1 per box)
classes_text = [] # List of string class name of bounding box (1 per box)
classes = [] # List of integer class id of bounding box (1 per box)
tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
}))
return tf_label_and_data

def main(_):
writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

file_loc = None
all_data_and_label_info = LOAD(file_loc)
# TODO END

for data_and_label_info in all_data_and_label_info:
tf_example = create_tf_example(data_and_label_info)
writer.write(tf_example.SerializeToString())

writer.close()

if __name__ == '__main__':
tf.app.run()