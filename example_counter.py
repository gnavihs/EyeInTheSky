exec(open("./File_Paths.py").read())    

print("AFTER MOVING:")
train_counter    = 0
test_counter     = 0

for root, di, files in os.walk(train_car_path):
    file_names = [os.path.join(train_car_path, f) for f in os.listdir(train_car_path) if os.path.isfile(os.path.join(train_car_path, f))]
    train_counter += len(file_names)

for root, di, files in os.walk(train_not_car_path):
    file_names = [os.path.join(train_not_car_path, f) for f in os.listdir(train_not_car_path) if os.path.isfile(os.path.join(train_not_car_path, f))]
    train_counter += len(file_names)

for root, di, files in os.walk(test_car_path):
    file_names = [os.path.join(test_car_path, f) for f in os.listdir(test_car_path) if os.path.isfile(os.path.join(test_car_path, f))]
    test_counter += len(file_names)

for root, di, files in os.walk(test_not_car_path):
    file_names = [os.path.join(test_not_car_path, f) for f in os.listdir(test_not_car_path) if os.path.isfile(os.path.join(test_not_car_path, f))]
    test_counter += len(file_names)

print("No. of training images:\t", train_counter)
print("No. of testing images:\t", test_counter)