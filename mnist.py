import tensorflow as tf
import tensorflow_datasets as tfds

# code paritally adapted from https://www.tensorflow.org/datasets/keras_example

batch_size=128
learning_rate=0.001
epochs=10

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Input((28, 28), name="mnist_in"),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation="softmax", name="mnist_out")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test,
)

train_loss=history.history['loss'][-1]
train_acc=history.history['sparse_categorical_accuracy'][-1]
val_loss=history.history['val_loss'][-1]
val_acc=history.history['val_sparse_categorical_accuracy'][-1]

print("train_loss: ", train_loss)
print("train_accuracy: ", train_acc)
print("val_loss: ", val_loss)
print("val_accuracy: ", val_acc)

tf.keras.models.save_model(model, "./model")
