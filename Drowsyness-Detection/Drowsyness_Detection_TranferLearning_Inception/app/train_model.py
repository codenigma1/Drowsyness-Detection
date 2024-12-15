import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.list_physical_devices('GPU')

batch_size = 16


train_data_generator = ImageDataGenerator(rescale=1./255, rotation_range=0.2, shear_range=0.2, zoom_range=0.2, 
                                          width_shift_range=0.2, height_shift_range=0.2, validation_split=0.2,horizontal_flip=True)



train_data = train_data_generator.flow_from_directory('data/train', target_size=(80, 80), 
                                                      batch_size=batch_size, class_mode='categorical', subset='training')
val_data = train_data_generator.flow_from_directory('data/train', target_size=(80, 80), 
                                                    batch_size=batch_size, class_mode='categorical', subset='validation')


test_data_generator = ImageDataGenerator(rescale=1./255)
test_data = test_data_generator.flow_from_directory('data/test', target_size=(80, 80), 
                                                    batch_size=batch_size, class_mode='categorical')

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(80, 80, 3))
# Access the output of the base model
head_model = base_model.output
# Adding a fully connected layer of finetuning
head_model = Flatten()(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(64, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

model = Model(inputs=base_model.input, outputs=head_model)
for layer in base_model.layers:
    layer.trainable = False


model.summary()

# Model Tranning Strategy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Save the best model as best_model.h5
checkpoint = ModelCheckpoint(
    filepath='save_models/new_best_model.h5',  # Save the best model as best_model.h5
    monitor='val_loss',       # Monitor validation loss
    verbose=3,                # Verbose output to show progress
    save_best_only=True,      # Save only the best model (not every epoch)
    save_weights_only=False   # Save the entire model, not just the weights
)
earlystop = EarlyStopping(monitor='val_loss', 
                          patience=7, 
                          verbose=3, 
                          restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            verbose=3)

callbacks = [checkpoint, earlystop, learning_rate_reduction]


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size,
    callbacks=callbacks,
    epochs=30
)
