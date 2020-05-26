import tensorflow as tf
import numpy as np

nb_classes = 5  # 1 ~ 3

X = tf.placeholder(tf.float32, shape = [None,1])
Y = tf.placeholder(tf.int32, shape = [None,1])
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([1, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

saver = tf.train.Saver()
model = tf.global_variables_initializer()

servo = float(input('servo: '))
# servo = 160
x_data = np.array([[servo/10]])
print(x_data)
with tf.Session() as sess:
    sess.run(model)
    save_path = "./pingpongsaved.cpkt"
    # save_path = "./softsaved.cpkt"
    saver.restore(sess,save_path)

    print(x_data)
    print(type(x_data))
    prediction = tf.argmax(hypothesis, 1)
    dict = sess.run(prediction,feed_dict={X:x_data})
    print(dict[0]+1)