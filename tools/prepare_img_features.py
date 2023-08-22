import cv2
import json
import numpy as np
import tensorflow as tf
import gc
from absl import flags
import logging
from keras.applications.resnet import ResNet50
import os
import sys
import psutil

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
        data = json.loads(fp.read())
    return data

# @tf.function
def main(argv):
    logging.basicConfig(level=logging.INFO)

    flags.DEFINE_string('action_reason_annot_path',
                        'data/train/QA_Combined_Action_Reason_train.json',
                        'Path to the action-reason annotation file.')

    flags.DEFINE_string('feature_extractor_name', 'inception_v4',
                        'The name of the feature extractor.')

    flags.DEFINE_string('feature_extractor_scope', 'InceptionV4',
                        'The variable scope of the feature extractor.')

    flags.DEFINE_string('feature_extractor_endpoint', 'PreLogitsFlatten',
                        'The endpoint of the feature extractor.')

    flags.DEFINE_string('feature_extractor_checkpoint', 'zoo/inception_v4.ckpt',
                        'The path to the checkpoint file.')

    flags.DEFINE_string('image_dir', 'data/train_images',
                        'Path to the ads image directory.')

    flags.DEFINE_string('output_feature_path', 'output/img_features_train.npy',
                        'Path to the output npy file.')

    flags.DEFINE_integer('batch_size', 32, 'The batch size.')
    
    # Replace flags.FLAGS(sys.argv) with FLAGS = flags.FLAGS(sys.argv)
    FLAGS = flags.FLAGS
    FLAGS(argv)

    examples = _load_annots(FLAGS.action_reason_annot_path)
    logging.info('Loaded %s examples.', len(examples))

    def _load_image(image_path, default_image_size):
        try:
            if not os.path.exists(image_path):
                logging.warning(f"Image file not found at {image_path}. Skipping...")
                return None
            with open(image_path, 'rb') as f:
                image_data = f.read()
            nparr = np.frombuffer(image_data, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None or bgr.size == 0:
                logging.warning(f"Error: Unable to read image from {image_path}.")
                return None
                
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) * 2.0 / 255.0 - 1.0
            rgb = cv2.resize(rgb, (default_image_size, default_image_size))
            return rgb
        except Exception as e:
            logging.error(str(e))
            return None

    skipped = 0
    
    def process_batch(batch_examples, model):
        nonlocal skipped
        batch_images = []
        image_ids = []
        for image_id, example in batch_examples:
            # filename = f"{FLAGS.image_dir}/{image_id}"
            filename = os.path.join(FLAGS.image_dir,image_id)
            # print(filename)
            rgb = _load_image(filename, default_image_size)
            if rgb is None:
                skipped+=1
                continue

            batch_images.append(rgb)
            image_ids.append(image_id)

        if batch_images:
            batch_images = np.stack(batch_images, axis=0)
            features = model.predict(batch_images)
            for img_id, feature in zip(image_ids, features):
                results[img_id] = feature

    # Create computational graph.
    g = tf.Graph()
    with g.as_default():
        # Create model.
        net_fn = ResNet50(weights=None, classes=1001)
        default_image_size = 224

        images = tf.keras.Input(shape=(default_image_size, default_image_size, 3), dtype=tf.float32)

        end_points = net_fn(images, training=False)

        output_tensor = tf.keras.Model(inputs=images, outputs=end_points)

        checkpoint = tf.train.Checkpoint(model=output_tensor)
        checkpoint.restore(FLAGS.feature_extractor_checkpoint)

        results = {}

        total_images = len(examples)
        processed_images = 0
        for batch_start in range(0, len(examples), FLAGS.batch_size):
            batch_examples = list(examples.items())[batch_start:batch_start + FLAGS.batch_size]
            process_batch(batch_examples, output_tensor)
            gc.collect()  # Perform garbage collection to release memory

            ram_usage = round(psutil.virtual_memory().used/(1024*1024*1024),2)
            disk_usage = psutil.disk_usage('/').percent
            
            processed_images += len(batch_examples)
            logging.info('Processed %d/%d images. Skipped %d RAM %d Disk %d', processed_images, total_images, skipped, ram_usage, disk_usage)

    # assert len(results) == len(examples)
    with open(FLAGS.output_feature_path, 'wb') as fp:
        np.save(fp, results)
    logging.info('Exported features for %i images.', len(results))
    logging.info('Done')


if __name__ == '__main__':
    main(sys.argv)
