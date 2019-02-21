import math

import matplotlib.pyplot as plt
import numpy as np

from utils import save


class Trainer():
    def __init__(self, sess, model, data_train, data_test=None, data_adv=None, batch_size=100):

        self.sess = sess
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.data_adv = data_adv
        self.batch_size = batch_size
        self.stats = {'Loss': [], 'Train': [], 'Test': [], 'Adv': []}

    def train(self, n_epochs, p_epochs=10):

        for epoch in range(n_epochs):

            train_loss, train_acc, test_acc, adv_acc = self.train_epoch(epoch)

            if (epoch + 1) % p_epochs == 0:
                print(
                    'Epoch : {:<3d} | Loss : {:.5f} | Accuracy : {:.5f} | Test Accuracy : {:.5f} | Adv Accuracy : {:.5f}'.format(
                        epoch + 1, train_loss, train_acc, test_acc, adv_acc))

            self.stats['Loss'].append(train_loss)
            self.stats['Train'].append(train_acc)
            self.stats['Test'].append(test_acc)
            self.stats['Adv'].append(adv_acc)

        save(self.stats, self.model.logdir + 'stats.pickle')

    def train_epoch(self, epoch):

        avg_loss = 0
        avg_acc = 0
        n_itrs = math.ceil(len(self.data_train[0]) / self.batch_size)

        for itr in range(n_itrs):
            loss, acc = self.train_step(itr)
            avg_loss += loss / n_itrs
            avg_acc += acc / n_itrs

        test_acc = self.model.evaluate(self.sess, self.data_test) if self.data_test else np.NaN
        adv_acc = self.model.evaluate(self.sess, self.data_adv) if self.data_adv else np.NaN

        self.model.save(self.sess)

        return avg_loss, avg_acc, test_acc, adv_acc

    def train_step(self, itr):

        batch_xs, batch_ys = np.copy(self.data_train[0][itr * self.batch_size:(itr + 1) * self.batch_size]), \
                             self.data_train[1][itr * self.batch_size:(itr + 1) * self.batch_size]
        feed_dict = {self.model.X: batch_xs, self.model.Y: batch_ys, self.model.training: True}
        _, loss, acc = self.sess.run([self.model.train, self.model.loss, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc


class VAE_GAN_Trainer():
    def __init__(self, sess, model, data_train, n_dis=1, batch_size=100):

        self.sess = sess
        self.model = model
        self.data_train = data_train
        self.n_dis = n_dis
        self.batch_size = batch_size
        self.stats = {'Enc Loss': [], 'Dec Loss': [], 'Disc Loss': []}

    def __visualize(self, images, title):

        plt.figure(figsize=(4, 4))

        for i in range(16):

            plt.subplot(4, 4, i + 1)

            if images.shape[-1] == 1:
                plt.imshow(images[i].reshape(28, 28), cmap='gray')
            else:
                plt.imshow(images[i].reshape(images.shape[1:]))

            plt.xticks([])
            plt.yticks([])

        plt.suptitle(title, fontsize=20, y=1.03)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        plt.show()
        plt.close()

    def train(self, n_epochs, p_epochs=10):

        for epoch in range(n_epochs):

            losses = self.train_epoch(epoch)

            if (epoch + 1) % p_epochs == 0:
                print('Epoch : {:<3d} | Enc Loss : {:.5f} | Dec Loss : {:.5f} | Disc Loss : {:.5f}'.format(epoch + 1,
                                                                                                           losses[0],
                                                                                                           losses[1],
                                                                                                           losses[2]))

                recs = (self.model.reconstruct(self.sess, (self.data_train[0][:16], self.data_train[1][:16])) + 1) * 0.5
                gens = (self.model.generate(self.sess, 16) + 1) * 0.5

                self.__visualize(recs, 'Reconstructions')
                self.__visualize(gens, 'Generations')

            self.stats['Enc Loss'].append(losses[0])
            self.stats['Dec Loss'].append(losses[1])
            self.stats['Disc Loss'].append(losses[2])

        save(self.stats, self.model.logdir + 'stats.pickle')

    def train_epoch(self, epoch):

        avg_enc_loss = 0
        avg_dec_loss = 0
        avg_disc_loss = 0
        n_itrs = round(len(self.data_train[0]) / self.batch_size)

        for itr in range(n_itrs):
            losses = self.train_step(itr)
            avg_enc_loss += losses[0] / n_itrs
            avg_dec_loss += losses[1] / n_itrs
            avg_disc_loss += losses[2] / n_itrs

        self.model.save(self.sess)

        return avg_enc_loss, avg_dec_loss, avg_disc_loss

    def train_step(self, itr):

        batch_xs, batch_ys = self.data_train[0][itr * self.batch_size:(itr + 1) * self.batch_size], self.data_train[1][
                                                                                                    itr * self.batch_size:(
                                                                                                                          itr + 1) * self.batch_size]

        feed_dict = {self.model.X: batch_xs, self.model.training: True}
        _, enc_loss = self.sess.run([self.model.enc_train, self.model.enc_loss], feed_dict=feed_dict)
        _, dec_loss = self.sess.run([self.model.dec_train, self.model.dec_loss], feed_dict=feed_dict)

        for _ in range(self.n_dis):
            _, disc_loss = self.sess.run([self.model.disc_train, self.model.disc_loss], feed_dict=feed_dict)

        return (enc_loss, dec_loss, disc_loss)
