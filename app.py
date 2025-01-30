import tensorflow as tf
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import threading
import os
import numpy as np
import logging
import time
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
import gc
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 5
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
class WebCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(WebCallback, self).__init__()
        self.epoch_times = []
        self.start_time = None
        self.epoch_accuracies = []  # Changed to store epoch accuracies
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        socketio.emit('training_start', {
            'time': self.start_time,
            'totalEpochs': self.params['epochs']
        })
        logger.info("Training started")
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch + 1}")
    def on_epoch_end(self, epoch, logs=None):
        try:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            current_metrics = {
                'epoch': epoch + 1,
                'accuracy': float(logs.get('accuracy', 0)),
                'loss': float(logs.get('loss', 0)),
            }
            socketio.emit('training_update', current_metrics)
            logger.info(f"Epoch {epoch + 1} completed. Accuracy: {current_metrics['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"Error in epoch_end: {str(e)}")
def create_improved_model():
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(2, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)
def create_dataset(image_paths, labels):
    def load_and_preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img / 255.0
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        lambda x, y: (load_and_preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
def evaluate_model(model, test_dataset, test_paths):
    predictions = []  
    actual = []      
    for images, labels in test_dataset.take(1):
        batch_predictions = model.predict(images)
        predictions.extend(batch_predictions)
        actual.extend(labels.numpy())
    test_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(actual, axis=1))
    precision = np.sum(np.logical_and(np.argmax(predictions, axis=1) == 1, np.argmax(actual, axis=1) == 1)) / np.sum(np.argmax(predictions, axis=1) == 1)
    recall = np.sum(np.logical_and(np.argmax(predictions, axis=1) == 1, np.argmax(actual, axis=1) == 1)) / np.sum(np.argmax(actual, axis=1) == 1)
    return {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall)
    }
def train_model():
    try:
        TRAIN_DIR = 'D:/Projects/Images_Classifier/Image-Classifier-ConvNet-in-Cats-Dog-dataset-master/train'
        TEST_DIR = 'D:/Projects/Images_Classifier/Image-Classifier-ConvNet-in-Cats-Dog-dataset-master/test1'
        train_paths, train_labels = [], []
        test_paths, test_labels = [], []
        logger.info("Preparing training data...")
        for img in os.listdir(TRAIN_DIR):
            label = [1, 0] if 'cat' in img else [0, 1]
            train_paths.append(os.path.join(TRAIN_DIR, img))
            train_labels.append(label)
        logger.info("Preparing test data...")
        for img in os.listdir(TEST_DIR):
            label = [1, 0] if 'cat' in img else [0, 1]
            test_paths.append(os.path.join(TEST_DIR, img))
            test_labels.append(label)

        split_idx = int(0.9 * len(train_paths))
        val_paths = train_paths[split_idx:]
        val_labels = train_labels[split_idx:]
        train_paths = train_paths[:split_idx]
        train_labels = train_labels[:split_idx]
        train_dataset = create_dataset(train_paths, train_labels)
        val_dataset = create_dataset(val_paths, val_labels)
        test_dataset = create_dataset(test_paths, test_labels)
        logger.info("Creating model...")
        model = create_improved_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Starting training...")
        callback = WebCallback()
        start_time = time.time()
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=[callback]
        )
        training_time = time.time() - start_time
        logger.info("Testing model...")
        test_loss, test_accuracy = model.evaluate(test_dataset)
        model.save(f'dogsvscats-improved-{time.strftime("%Y%m%d-%H%M%S")}.keras')
        evaluation_results = evaluate_model(model, test_dataset, test_paths)
        socketio.emit('training_complete', {
            'trainAccuracy': float(history.history['accuracy'][-1]),
            'testAccuracy': float(evaluation_results['test_accuracy']),
            'testPrecision': float(evaluation_results['test_precision']),
            'testRecall': float(evaluation_results['test_recall'])
        })

    except Exception as e:
        error_msg = f"Error in training: {str(e)}"
        logger.error(error_msg)
        socketio.emit('training_error', {'error': error_msg})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    return True

@socketio.on('disconnect')
def handle_disconnect():
    return True

if __name__ == '__main__':
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()

    try:
        socketio.run(app, debug=True, port=5000)
    except Exception as e:
        logger.error(f"Error running SocketIO server: {e}")