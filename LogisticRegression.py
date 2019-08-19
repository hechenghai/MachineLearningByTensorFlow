import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def LogisticRegression(learning_rate,training_epoch,batch_size,scale):
    
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])
    regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    weights = tf.get_variable(name="weights",shape=[784,10],regularizer=regularizer)
    biases = tf.Variable(tf.zeros([10]),name='biases')
    pred = tf.nn.sigmoid(tf.matmul(x,weights)+biases)
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),axis=1))
    keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    l2_cost = cost + tf.add_n(keys)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epoch):
            average_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for batch in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
                average_cost += c/total_batch
            if epoch % 2 == 0:
                print(epoch,average_cost)
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print("Accuracy = " + str(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})))

if __name__ == '__main__':

    LogisticRegression(0.1,1000,200,0.001)
