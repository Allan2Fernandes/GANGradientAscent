import tensorflow as tf

class DatasetBuilder:
    def __init__(self, directory_path, batch_size, target_size):
        self.directory_path = directory_path
        self.batch_size = batch_size
        self.target_size = target_size

        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.directory_path,
            labels=None,
            label_mode=None,
            class_names=None,
            color_mode='rgb',
            batch_size=None,
            image_size=self.target_size,
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=True)

        self.image_dataset = dataset.map(self.map_dataset).batch(batch_size=batch_size, drop_remainder=True).prefetch(buffer_size=1)
        pass

    def map_dataset(self, datapoint):
        datapoint = (datapoint-127.5)/127.5
        return datapoint

    def get_dataset(self):
        return self.image_dataset

    pass