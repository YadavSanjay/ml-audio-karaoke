import tensorflow as tf
import argparse
import numpy as np
import time
import logging
import os

from com.syml.learn.music.karaoke.medleydb_analyzer import MedleyDBAnalyser

N_BINS  = 1025
N_SAMPLES = 20
VECTOR_SIZE=N_BINS * N_SAMPLES

logs_path = "D:/tmp/tensorflow_logs/music"
destDir = 'D:/Tools/ml/music/karaoke/masks/'
rootDir = 'D:/Tools/store/medleydb/MedleyDB/Audio/'

class VocalSeparator:

    def __init__(self):
        print('init VocalSeparator... ')

    def variable_summaries(self,var,name):
        tf.summary.scalar(name, var);
        return

    def _weightVariable(self,shape,vname):
        initial = tf.truncated_normal(shape, stddev=0.007)
        return tf.Variable(initial,name=vname)

    def _biasVariable(self,shape,constVal,vname):
        initial = tf.constant(constVal, shape=shape)
        return tf.Variable(initial,name=vname)

    def buildModel(self):
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, VECTOR_SIZE], name="x-input")
            z_ = tf.placeholder(tf.float32, shape=[None, VECTOR_SIZE], name="z-output")
            self.x = x
            self.z_ = z_

        with tf.name_scope("weights"):
            Wxy = self._weightVariable([VECTOR_SIZE, VECTOR_SIZE],"w1")
            Wyz = self._weightVariable([VECTOR_SIZE, VECTOR_SIZE],"w2")

        with tf.name_scope("biases"):
            by = self._biasVariable([VECTOR_SIZE],0.005,"b1")

        with tf.name_scope("model"):
            y0 = tf.matmul(x, Wxy) + by;
            y = tf.nn.sigmoid(y0,"hidden")
            z  = tf.matmul(y, Wyz)
            self.z = z
            tf.summary.histogram("weights1", Wxy)
            tf.summary.histogram("biases1", by)
            tf.summary.histogram("weights2", Wyz)

        with tf.name_scope('cost'):
            self._cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=z_, logits=z))
            # regularizer = tf.nn.l2_loss(Wxy) + tf.nn.l2_loss(Wyz)
            # self._cost =  self._cross_entropy + 0.001* regularizer
            self.variable_summaries(self._cross_entropy,"xentropy")

        with tf.name_scope('Accuracy'):
            p1 = tf.sigmoid(z)
            p = tf.round(p1)
            pnorm = tf.nn.l2_normalize(p,1)
            labelnorm = tf.nn.l2_normalize(z_, 1)
            cos_similarity = tf.reduce_sum(tf.multiply(pnorm,labelnorm),1)
            tf.summary.histogram("cos_similarity", cos_similarity)
            self._accuracy_1 = cos_similarity
            # self._distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z_, p1)), 1))
            # tf.summary.histogram("distance", self._distance)

        learning_rate = 1.0e-4
        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(self._cross_entropy);
            self._train_op=train_op
        self.summ = tf.summary.merge_all()

    def train(self):
        medley = MedleyDBAnalyser(rootDir=rootDir, destDir=destDir)
        sess = tf.Session()
        writer = tf.summary.FileWriter(logs_path)
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        NUM_EPOCHS = 20
        BATCH_SIZE = 20
        j = 0
        for epoch in range(0,NUM_EPOCHS):
            i=0
            batches = medley.getSampleBatch(batchSize=BATCH_SIZE)
            for batch in batches:
                if i == 0:
                    start_time = time.time()
                if i % 10 == 0:
                    [train_accuracy, s] = sess.run([self._accuracy_1, self.summ], feed_dict={self.x: batch[0], self.z_: batch[1]})
                    writer.add_summary(s, j)
                    j=j+1
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    logging.info('epoch=' + str(epoch) + ', iteration='+ str(i) + ', samples=' + str(i*BATCH_SIZE) + ', time[s]=' + str(elapsed_time))
                    start_time = time.time()
                sess.run(self._train_op,feed_dict={self.x: batch[0], self.z_: batch[1]})
                i += 1
            logging.info('end of epoch...'+ str(epoch))
        saver.save(sess, os.path.join(logs_path, "karaoke_model.ckpt"))
        logging.info('end of training...')


def main(_):
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    v = VocalSeparator()
    v.buildModel()
    v.train()



if __name__ == '__main__':

    import sys
    print(sys.version)
    tensorboard_log_dir = '/path/to/tensorboard/log'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')

    parser.add_argument(
        '--log_dir',
        type=str,
        default=tensorboard_log_dir,
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
