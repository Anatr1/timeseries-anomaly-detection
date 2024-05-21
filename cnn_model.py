import os
import tensorflow as tf
import keras_tuner as kt

class CNNClassifier:
    def __init__(self, input_shape, num_classes, max_epochs=50, output_dir='./models_output'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.tuner = None
        self.model = None
        
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            print(f"Directory {output_dir} already exists.")
    
    def build_model(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        model.add(tf.keras.layers.Reshape((self.input_shape[0], 1)))
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(tf.keras.layers.Conv1D(
                filters=hp.Int('units_' + str(i), 32, 256, 32),
                kernel_size=hp.Int('kernel_size_' + str(i), 3, 5),
                activation='relu',
                padding='same'
            ))
            model.add(tf.keras.layers.MaxPooling1D(
                pool_size=hp.Int('pool_size_' + str(i), 2, 4),
                padding='same'
            ))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model
    
    def setup_tuner(self):
        self.tuner = kt.Hyperband(
            self.build_model,
            overwrite=True,
            objective='accuracy',
            max_epochs=self.max_epochs,
            factor=3,
            directory=os.path.join(self.output_dir, "keras_tuner_cnn"),
            project_name="keras_tuner_cnn_prj"
        )
    
    def search(self, X_train, y_train):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15)
        
        self.tuner.search(X_train, y_train, epochs=self.max_epochs, validation_split=0.2,
                          callbacks=[early_stopping,
                                     tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.output_dir, "/tmp/tb_logs"))])
    
        self.model = self.tuner.get_best_models()[0]
    
    def predict(self, X_test):
        if self.model is None:
            raise Exception("Model is not trained yet. Call the search method to train the model.")
        return self.model.predict(X_test)