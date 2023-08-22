import cv2
import json
import numpy as np
import tensorflow as tf
import gc

from keras.applications.resnet import ResNet50

from absl import flags
import logging
import psutil

import os
import sys

# path = 'path to ADVISE'
# os.chdir(path)
# sys.path.append(path)

from utils.train_utils import default_session_config

def _load_annots(filename):
    """Loads annotation file.

    Args:
      filename: path to the annotation json file.

    Returns:
      examples: a dict mapping from img_id to annotation.
    """
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def image_crop_and_resize(image, bbox, crop_size):
    """Crops roi from an image and resizes it.

    Args:
      image: the image.
      bbox: bounding box information.
      crop_size: the expected output size of the roi image.

    Returns:
      a [crop_size, crop_size, 3] roi image.
    """
    height, width, _ = image.shape

    x1, y1, x2, y2 = bbox
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)

    return cv2.resize(image[y1: y2, x1: x2, :], crop_size)

def main(argv):
    logging.basicConfig(level=logging.INFO)

    # Initialize flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('bounding_box_json', '',
                        'Path to the json annotation file.')

    flags.DEFINE_string('feature_extractor_checkpoint', 'zoo/inception_v4.ckpt',
                        'The path to the checkpoint file.')

    flags.DEFINE_string('image_dir', 'data/train_images/',
                        'Path to the ads image directory.')

    flags.DEFINE_string('output_feature_path', 'output/roi_features.npy',
                        'Path to the output npy file.')

    flags.DEFINE_integer('max_number_of_regions', 10,
                         'Maximum number of regions.')

    FLAGS(argv)

    examples = _load_annots(FLAGS.bounding_box_json)
    logging.info('Loaded %s examples.', len(examples))

    def _load_image(image_path, default_image_size):
        try:
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None or bgr.size == 0:
                raise Exception(f"Error: Unable to read image from {image_path}.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) * 2.0 / 255.0 - 1.0
            rgb = cv2.resize(rgb, (default_image_size, default_image_size))
            return rgb
        except Exception as e:
            logging.error(str(e))
            return None

    # Create model
    default_image_size = 224

    images = tf.keras.Input(shape=(default_image_size, default_image_size, 3), dtype=tf.float32)
    net_fn = ResNet50(include_top=False, weights=None, input_tensor=images, pooling='avg')

    # Load checkpoint
    checkpoint = tf.train.Checkpoint(model=net_fn)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=FLAGS.feature_extractor_checkpoint, max_to_keep=1)
    checkpoint_manager.restore_or_initialize()

    results = {}

    batch_size = 16
    for batch_start in range(0, len(examples), batch_size):
        batch_examples = list(examples.items())[batch_start:batch_start + batch_size]
        batch_results = {}

        for index, (image_id, example) in enumerate(batch_examples):
            ram_usage = round(psutil.virtual_memory().used/(1024*1024*1024),5)
            disk_usage = psutil.disk_usage('/').percent
            logging.info('On image %i/%i. RAM %d DISK %d', index + batch_start, len(examples), ram_usage, disk_usage)

            filename = os.path.join(FLAGS.image_dir, image_id)
            rgb = _load_image(filename, default_image_size)
            if rgb is None:
                continue

            regions_list = []
            for region in example['regions'][:FLAGS.max_number_of_regions]:
                roi = image_crop_and_resize(rgb,
                                            bbox=(
                                                region['bbox']['xmin'],
                                                region['bbox']['ymin'],
                                                region['bbox']['xmax'],
                                                region['bbox']['ymax']),
                                            crop_size=(default_image_size, default_image_size))
                regions_list.append(roi)
            batch_results[image_id] = regions_list

        batch = np.stack([roi for regions_list in batch_results.values() for roi in regions_list], axis=0)
        features = net_fn.predict(batch)
        feature_index = 0
        for image_id, regions_list in batch_results.items():
            num_regions = len(regions_list)
            results[image_id] = features[feature_index:feature_index + num_regions]
            feature_index += num_regions

        del rgb, regions_list, batch, features, roi
        gc.collect()

    # Write results
    with open(FLAGS.output_feature_path, 'wb') as fp:
        np.save(fp, results)
    logging.info('Exported features for %i images.', len(results))

    logging.info('Done')

if __name__ == '__main__':
    # Pass sys.argv to the main function for proper argument parsing
    main(sys.argv)
