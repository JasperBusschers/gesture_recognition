
import tensorflow as tf




def get_label(file_path):
  # convert the path to a list of path components
  file_path = tf.strings.regex_replace(file_path, 'jpg', 'png')
  #label = tf.io.read_file(file_path)
  label = tf.io.read_file(file_path)
  label = tf.image.decode_png(label)
  label = tf.image.central_crop(label, 0.8)
  label = tf.image.convert_image_dtype(label, tf.float32)
  label = tf.image.resize(label, [32, 32])
  return label


def decode_img(img_path):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.io.read_file(img_path)
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.central_crop(img, 0.8)
  # resize the image to the desired size.
  return tf.image.resize(img, [32, 32])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(file_path)
  return img, label

def load_dataset(path, batch_size):
    files = tf.data.Dataset.list_files(path, shuffle=True)
    train_size = round(0.75 * len(list(files)))
    test_size = len(list(files)) - train_size
    print("dataset size is :  " + str(files.__sizeof__()) + " test_size =  " + str(test_size))
    test_dataset = files.take(test_size)
    train_dataset = files.skip(test_size)
    labeled_ds = train_dataset.map(process_path, num_parallel_calls=4).batch(batch_size)
    labeled_ds_test = test_dataset.map(process_path, num_parallel_calls=4).batch(batch_size)
    return labeled_ds, labeled_ds_test

