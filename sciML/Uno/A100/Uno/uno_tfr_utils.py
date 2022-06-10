import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_feature(feature, label):
  #define the dictionary -- the structure -- of our single example
  data = {
        'feature_len': _int64_feature(len(feature)),
        'feature'   : _bytes_feature(serialize_array(feature)),
        'label'     : _float_feature(label)
    }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out

def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'feature_len' : tf.io.FixedLenFeature([], tf.int64),
      'feature'   : tf.io.FixedLenFeature([], tf.string),
      'label' : tf.io.FixedLenFeature([], tf.float32),
    }

  content = tf.io.parse_single_example(element, data)
  
  feature_len = content['feature_len']
  feature_raw  = content['feature']
  feature = tf.io.parse_tensor(feature_raw, out_type=tf.float32)
  feature = tf.reshape(feature, shape=[feature_len])
  label = content['label']
  return (feature, label)

def write_feature_to_tfr_short(feature, labels, filename:str="features.tfrecords"):
  writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
  count = 0

  rows, cols = feature.shape
  for index in range(rows):

    #get the data we want to write
    current_feat = feature[index,:] 
    current_label = labels[index]

    out = parse_single_feature( current_feat, current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count
