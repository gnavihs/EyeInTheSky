exec(open("./inception_v3.py").read())


def preprocess_input(x):
    x = np.true_divide(x, 255)
    # x /= 255.
    x -= 0.5
    x *= 2.
    return x

def crop_center_train(img):
    img     = preprocess_input(img)
    cropx   = img_size
    cropy   = img_size
    y,x,c   = img.shape
    x_shift = randint(-12,12)
    y_shift = randint(-12,12)

    img_final = np.zeros((original_img_size, original_img_size, 3), dtype=np.float32)
    diff = original_img_size - img_size
    margin = diff//2

    startx = x//2 - cropx//2 + x_shift
    starty = y//2 - cropy//2 + y_shift  
    img_final[margin:margin+img_size, margin:margin+img_size, :] = img[starty:starty+cropy, startx:startx+cropx, :]
    return img_final

def crop_center_test(img):
    img     = preprocess_input(img)
    cropx   = img_size
    cropy   = img_size
    y,x,c   = img.shape

    img_final = np.zeros((original_img_size, original_img_size, 3), dtype=np.float32)
    diff = original_img_size - img_size
    margin = diff//2

    startx = x//2 - cropx//2
    starty = y//2 - cropy//2  
    img_final[margin:margin+img_size, margin:margin+img_size, :] = img[starty:starty+cropy, startx:startx+cropx, :]
    return img_final

if __name__ == '__main__':

    #Do NOT change these
    img_rows    = original_img_size
    img_cols    = original_img_size # Resolution of inputs
    channel     = 3

    num_classes = 2
    batch_size  = 64 
    nb_epoch    = 30
##########################################################################################################################################################################################
# this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # zca_whitening=False,
    # zca_epsilon=1e-6,
    # rotation_range=0.,
    # width_shift_range=0.046875,
    # height_shift_range=0.046875,
    # shear_range=0.,
    # zoom_range=0.,
    # channel_shift_range=0.,
    # fill_mode='nearest',
    # cval=0.,
    # horizontal_flip=False,
    # vertical_flip=False,
    rescale=None,
    preprocessing_function=crop_center_train,
    data_format=K.image_data_format()
    )

    train_generator = train_datagen.flow_from_directory(
    os.path.join(patch_sets_path, "data/train"),
    target_size=(256, 256),
    batch_size=batch_size,
    classes=['car', 'not_car'],
    class_mode='categorical',
    )

    test_datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # zca_whitening=False,
    # zca_epsilon=1e-6,
    # rotation_range=0.,
    # width_shift_range=0.,
    # height_shift_range=0.,
    # shear_range=0.,
    # zoom_range=0.,
    # channel_shift_range=0.,
    # fill_mode='nearest',
    # cval=0.,
    # horizontal_flip=False,
    # vertical_flip=False,
    rescale=None,
    preprocessing_function=crop_center_test,
    data_format=K.image_data_format()
    )

    test_generator = test_datagen.flow_from_directory(
    os.path.join(patch_sets_path, "data/test"),
    target_size=(256, 256),
    batch_size=batch_size,
    classes=['car', 'not_car'],
    class_mode='categorical',
    )
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
##########################################################################################################################################################################################
    # FIRST STAGE: Training ONLY the last layer(softmax) of the updated inception model
    print("FIRST STAGE: Training ONLY the last layer (softmax) of the updated inception model")
    # Load our model
    model = InceptionV3(include_top=False, 
                        weights='imagenet', 
                        input_shape=(img_rows, img_cols, channel), 
                        pooling='avg',
                        trainable=False, 
                        classes=num_classes,
                        second_stage=False,
                        model_weights='imagenet_models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # model.summary()

    # Some callbacks for logging
    tensorboard     = TensorBoard(log_dir='./logs')
    early_stopping  = EarlyStopping(monitor='val_loss', patience=2)
    reduceLR        = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    checkpointer    = ModelCheckpoint(filepath='imagenet_models/my_model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # Start Fine-tuning
    # hist = model.fit(X_train, Y_train,
    #                   batch_size=batch_size,
    #                   epochs=nb_epoch,
    #                   shuffle=True,
    #                   verbose=1,
    #                   validation_data=(X_test, Y_test),
    #                   callbacks=[tensorboard, early_stopping, reduceLR, checkpointer]
    #                   )

    hist = model.fit_generator(
                            train_generator,
                            steps_per_epoch=train_counter // batch_size,
                            epochs=nb_epoch,
                            validation_data=test_generator,
                            validation_steps=test_counter // batch_size,
                            callbacks=[tensorboard, early_stopping, reduceLR, checkpointer]
                            )
    # print(hist.history)
    del model  # deletes the existing model
##########################################################################################################################################################################################
    # SECOND STAGE: Training all the layers of the updated inception model
    print("SECOND STAGE: Training all the layers of the updated inception model")

    # Load our model
    model = InceptionV3(include_top=False, 
                        weights='imagenet', 
                        input_shape=(img_rows, img_cols, channel), 
                        pooling='avg',
                        trainable=True,
                        classes=num_classes,
                        second_stage=True,
                        model_weights='imagenet_models/my_model.h5')

    # Some callbacks for logging
    early_stopping  = EarlyStopping(monitor='val_loss', patience=3)
    reduceLR        = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    checkpointer    = ModelCheckpoint(filepath='imagenet_models/my_model_final.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    # Start Fine-tuning
    # hist = model.fit(X_train, Y_train,
    #                   batch_size=batch_size,
    #                   epochs=nb_epoch,
    #                   shuffle=True,
    #                   verbose=1,
    #                   validation_data=(X_test, Y_test),
    #                   callbacks=[tensorboard, early_stopping, reduceLR, checkpointer]
    #                   )

    hist = model.fit_generator(
                            train_generator,
                            steps_per_epoch=train_counter // batch_size,
                            epochs=nb_epoch,
                            validation_data=test_generator,
                            validation_steps=test_counter // batch_size,
                            callbacks=[tensorboard, early_stopping, reduceLR, checkpointer]
                            )

    del model  # deletes the existing model
    
    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))