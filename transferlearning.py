import time
from datetime import timedelta
# We use Pretty Tensor to define the new classifier.
import prettytensor as pt
from numpy import array
#Read configuration
exec(open("./configuration.py").read())
#Read images from files
exec(open("./DataImport2.py").read())
'''
###########################################################################
######################### Data Import Session #############################
# dataset of patches and labels

sessionDI = tf.Session()
sessionDI.run(tf.global_variables_initializer())
sessionDI.close()

ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])

###########################################################################
'''
#'1' for car, '0' for not car
labels_train    = []
labels_test     = []

#Load labels for all the train images
cache_path = file_path_cache_train + 'labels' + cache_extension
# If the cache-file exists.
if os.path.exists(cache_path):
    # Load the cached data from the file.
    with open(cache_path, mode='rb') as file:
        labels_train = pickle.load(file)

#Converting labels into one hot array
labels_one_hot_train    = np.zeros((len(labels_train), max(labels_train)+1))
labels_one_hot_train[np.arange(len(labels_train)),labels_train] = 1
labels_train            = array(labels_train)

#Load labels for all the test images
cache_path = file_path_cache_test + 'labels' + cache_extension
# If the cache-file exists.
if os.path.exists(cache_path):
    # Load the cached data from the file.
    with open(cache_path, mode='rb') as file:
        labels_test = pickle.load(file)

#Converting labels into one hot array
labels_one_hot_test     = np.zeros((len(labels_test), max(labels_test)+1))
labels_one_hot_test[np.arange(len(labels_test)),labels_test] = 1
labels_test             = array(labels_test)
###########################################################################
############################New network####################################

transfer_len    = model.transfer_len
x               = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true          = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls      = tf.argmax(y_true, dimension=1)

# Wrap the transfer-values as a Pretty Tensor object.
x_pretty        = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        softmax_classifier(num_classes=num_classes, labels=y_true, name="softmax_classifiers")

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
optimizer           = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step, name='optimiser')
y_pred_cls          = tf.argmax(y_pred, dimension=1, name='y_pred_cls')
correct_prediction  = tf.equal(y_pred_cls, y_true_cls, name='correct_prediction')
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
###########################################################################
#############Add ops to save and restore all the variables#################
saver = tf.train.Saver()

###########################################################################
###############################Session#####################################
session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 256
def random_batch():

    cache_file_id = np.random.choice(total_train_cache_files,
                                    size=1,
                                    replace=False)    
    cache_path = file_path_cache_train + str(cache_file_id[0]) + cache_extension
    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            transfer_values_train = pickle.load(file)

    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=num_images if train_batch_size>num_images else train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    labels_id = [cache_file_id[0]*cache_batch_size + id for id in idx]
    y_batch = labels_one_hot_train[labels_id]

    # print("cache file:",cache_path)
    # print("labels_id:",labels_id)
    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        #IMPORTANT NOTE: Ideally accuracy on whole training set should be calculated
        #But not done here......To be discussed later
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy of latest batch: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Split the data-set in batches of this size to limit RAM usage.
test_batch_size = 256
def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
        
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test(transfer_values_test, index):
    labels_id_test = []
    # labels_id_test = range(cache_batch_size) if cache_batch_size<len(transfer_values_test) else range(len(transfer_values_test))
    labels_id_test = range(len(transfer_values_test))
    labels_id_test = [index*cache_batch_size + id for id in labels_id_test]
    # print("labels_id_test: ", labels_id_test)
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_one_hot_test[labels_id_test],
                       cls_true = labels_test[labels_id_test])


def classification_accuracy(correct):
    # Return the classification accuracy
    # and the number of correct classifications.
    return sum(correct) / float(len(correct)), sum(correct)


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    
    correct     = []
    cls_pred    = []

    for i in range(total_test_cache_files):
        cache_path = file_path_cache_test + str(i) + cache_extension
        # print("cache_path:", cache_path)
        if os.path.exists(cache_path):
        # Load the cached data from the file.
            with open(cache_path, mode='rb') as file:
                transfer_values_test = pickle.load(file)
                # For all the images in the test-set,
                # calculate the predicted classes and whether they are correct.
                correct_batch, cls_pred_batch = predict_cls_test(transfer_values_test, i)
                correct.extend(correct_batch)
                cls_pred.extend(cls_pred_batch)
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

'''
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
'''

#################################################################################################################
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)

optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)

save_path = saver.save(session, "./tmp/model")
print("Model saved in file: %s" % save_path)

for i in tf.get_default_graph().get_operations():
    print(i.values())
session.close()
