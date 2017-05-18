exec(open("./DataImport.py").read())
# exec(open("./inception.py").read())

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  # Start populating the filename queue.
  print("AAAAAAA")
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  # for i in range(1): #length of your filename list
  #   image = my_img.eval() #here is your image Tensor :)
  #   image_pos = my_img_pos.eval()
  #   image_neg = my_img_neg.eval()
  print("AAAAAAA")
  print(session.run(indices_pos))
  print(session.run(my_img))
  #images = pre_process_image(my_img, 1, indices.eval())
  # a = session.run(shapeOp)
  # print(a)
  print("AAAAAAA")
  
  coord.request_stop()
  coord.join(threads)
