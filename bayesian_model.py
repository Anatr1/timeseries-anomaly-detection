import os
import tensorflow as tf
import keras_tuner as kt

class BayesianMLPClassifier:
    def __init__(self, input_shape, num_classes, max_epochs=50, output_dir='./models'):
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
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        hp_drop_rate = hp.Float('drop_rate', min_value=0.1, max_value=0.75, step=0.05)
        hp_drop_rate_last = hp.Float('drop_rate', min_value=0.1, max_value=0.75, step=0.05)
        
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs
        for i in range(hp.Int("mlp_layers", 1, 3)):
            x = tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", 32, 128, step=32), activation="relu"
            )(x)
            x = tf.keras.layers.Dropout(rate=hp_drop_rate)(x, training=True)
        x = tf.keras.layers.Dropout(rate=0.5)(x, training=True)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            metrics=['accuracy']
        )
        
        return model

    def setup_tuner(self):
        self.tuner = kt.Hyperband(
            self.build_model,
            overwrite=True,
            objective='accuracy',
            max_epochs=self.max_epochs,
            factor=3,
            directory=os.path.join(self.output_dir, "keras_tuner"),
            project_name="keras_tuner_prj"
        )

    def search(self, X_train, y_train):
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=15)
        
        self.tuner.search(
            X_train, y_train, epochs=self.max_epochs,
            callbacks=[
                stop_early,
                tf.keras.callbacks.TensorBoard(os.path.join(self.output_dir, "/tmp/tb_logs"))
            ]
        )
        self.model = self.tuner.get_best_models()[0]

    def predict(self, X_test):
        if self.model is None:
            raise Exception("Model is not trained yet. Call the search method to train the model.")
        return self.model.predict(X_test)
