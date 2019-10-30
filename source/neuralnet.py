import tensorflow as tf
import numpy as np

class ConAD(object):

    def __init__(self, height, width, channel, z_dim, num_h, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.z_dim, self.num_h, self.leaning_rate = z_dim, num_h, leaning_rate
        self.k_size = 3

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.z_dim])
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []

        self.z_enc, self.z_mu, self.z_sigma, self.x_batbat, self.hypo_batbat = \
            self.build_autoencoder(input=self.x, ksize=self.k_size)

        self.restore_error_h = self.mean_square_error(x1=self.x_batbat, x2=self.hypo_batbat)
        self.best_idx = np.argmin(self.restore_error_h)
        self.x_best = self.hypo_batbat[self.best_idx]
        self.x_nonbest = self.hypo_batbat

        self.x_fake = self.hypotheses(input=self.decoder(input=self.z, ksize=self.k_size), \
            ksize=self.k_size, h_num=self.best_idx, expand=False)

        """Loss D"""
        self.d_real, self.d_fake, self.d_best, self.d_others = \
            self.build_discriminator(real=self.x, \
            fake=self.x_fake, best=self.x_best, others=self.x_nonbest, ksize=self.k_size)

        self.l_real = -tf.math.log(self.d_real + 1e-12)
        self.l_fake = (tf.math.log(self.d_fake + 1e-12) + tf.math.log(self.d_best + 1e-12) + tf.math.log(self.d_others + 1e-12)) / (self.num_h + 1)
        self.loss_d = tf.abs(tf.reduce_mean(self.l_real + self.l_fake))

        """Loss G"""
        self.restore_error = -tf.reduce_sum(self.x * tf.math.log(self.x_best + 1e-12) + (1 - self.x) * tf.math.log(1 - self.x_best + 1e-12), axis=(1, 2, 3)) # binary cross-entropy
        self.kl_divergence = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.math.log(tf.square(self.z_sigma) + 1e-12) - 1, axis=(1))

        self.mean_restore = tf.reduce_mean(self.restore_error)
        self.mean_kld = tf.reduce_mean(self.kl_divergence)
        self.ELBO = tf.reduce_mean(self.restore_error + self.kl_divergence) # Evidence LowerBOund
        self.loss_g = self.ELBO - self.loss_d

        """Vars"""
        self.vars1, self.vars2 = [], []
        for widx, wname in enumerate(self.w_names):
            if("dis" in wname):
                self.vars1.append(self.weights[widx])
                self.vars1.append(self.biasis[widx])
            elif(("enc" in wname) or ("dec" in wname) or ("hypo" in wname)):
                self.vars2.append(self.weights[widx])
                self.vars2.append(self.biasis[widx])
            else: pass

        print("\nVariables (D)")
        for var in self.vars1: print(var)
        print("\nVariables (G)")
        for var in self.vars2: print(var)

        """Optimizer"""
        self.optimizer_d = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss_d, var_list=self.vars1, name='Adam_D')
        self.optimizer_g = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss_g, var_list=self.vars2, name='Adam_G')

        self.mse_r = self.mean_square_error(x1=self.x, x2=self.x_best)

        tf.compat.v1.summary.scalar('Loss_D/d_real', tf.reduce_mean(self.d_real))
        tf.compat.v1.summary.scalar('Loss_D/d_fake', tf.reduce_mean(self.d_fake))
        tf.compat.v1.summary.scalar('Loss_D/d_best', tf.reduce_mean(self.d_best))
        tf.compat.v1.summary.scalar('Loss_D/d_others', tf.reduce_mean(self.d_others))
        tf.compat.v1.summary.scalar('Loss_D/loss_d', self.loss_d)

        tf.compat.v1.summary.scalar('Loss_G/mean_restore', self.mean_restore)
        tf.compat.v1.summary.scalar('Loss_G/mean_kld', self.mean_kld)
        tf.compat.v1.summary.scalar('Loss_G/loss_g', self.loss_g)

        self.summaries = tf.compat.v1.summary.merge_all()

    def mean_square_error(self, x1, x2):

        data_dim = len(x1.shape)
        if(data_dim == 5):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3, 4))
        elif(data_dim == 4):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif(data_dim == 3):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif(data_dim == 2):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1))
        else:
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2))

    def build_autoencoder(self, input, ksize=3):

        with tf.name_scope('encoder') as scope_enc:
            z_enc, z_mu, z_sigma = self.encoder(input=input, ksize=ksize)

        with tf.name_scope('decoder') as scope_enc:
            x_hat = self.decoder(input=z_enc, ksize=ksize)

            x_batbat, hypo_batbat = None, None
            for h in range(self.num_h):
                x_h = self.hypotheses(input=x_hat, ksize=ksize, h_num=h)
                if(x_batbat is None):
                    x_batbat = tf.expand_dims(input=input, axis=0)
                    hypo_batbat = x_h
                else:
                    x_batbat = tf.concat([x_batbat, tf.expand_dims(input=input, axis=0)], 0)
                    hypo_batbat = tf.concat([hypo_batbat, x_h], 0)

        return z_enc, z_mu, z_sigma, x_batbat, hypo_batbat

    def build_discriminator(self, real, fake, best, others, ksize=3):

        with tf.name_scope('discriminator') as scope_dis:
            d_real = self.discriminator(input=real, ksize=ksize)
            d_fake = self.discriminator(input=fake, ksize=ksize)
            d_best = self.discriminator(input=best, ksize=ksize)
            d_others = None
            for h in range(self.num_h):
                if(d_others is None):
                    d_others = self.discriminator(input=others[h], ksize=ksize)
                else:
                    d_others += self.discriminator(input=others[h], ksize=ksize)
            d_others /= self.num_h

        return d_real, d_fake, d_best, d_others

    def encoder(self, input, ksize=3):

        print("\nEncode-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="elu", name="enc1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="enc1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")
        self.conv_shapes.append(conv1_2.shape)

        print("Encode-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="enc2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="enc2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")
        self.conv_shapes.append(conv2_2.shape)

        print("Encode-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="enc3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="enc3_2")
        self.conv_shapes.append(conv3_2.shape)

        print("Dense (Fully-Connected)")
        self.fc_shapes.append(conv3_2.shape)
        [n, h, w, c] = self.fc_shapes[0]
        fulcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="fulcon_in")
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="elu", name="encfc_1")

        z_params = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=self.z_dim*2, activation="None", name="encfc_2")
        z_mu, z_sigma = self.split_z(z=z_params)
        z = self.sample_z(mu=z_mu, sigma=z_sigma) # reparameterization trick

        return z, z_mu, z_sigma

    def decoder(self, input, ksize=3):

        print("\nDecode-Dense")
        [n, h, w, c] = self.fc_shapes[0]
        fulcon2 = self.fully_connected(input=input, num_inputs=int(self.z_dim), \
            num_outputs=512, activation="elu", name="decfc_1")
        fulcon3 = self.fully_connected(input=fulcon2, num_inputs=int(fulcon2.shape[1]), \
            num_outputs=int(h*w*c), activation="elu", name="decfc_2")
        fulcon_out = tf.compat.v1.reshape(fulcon3, shape=[self.batch_size, h, w, c], name="decfc_3")

        print("Decode-1")
        convt1_1 = self.conv2d(input=fulcon_out, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="dec1_1")
        convt1_2 = self.conv2d(input=convt1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="dec1_2")

        print("Decode-2")
        [n, h, w, c] = self.conv_shapes[-2]
        convt2_1 = self.conv2d_transpose(input=convt1_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 32, 64], \
            dilations=[1, 1, 1, 1], activation="elu", name="dec2_1")
        convt2_2 = self.conv2d(input=convt2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="dec2_2")

        print("Decode-3")
        [n, h, w, c] = self.conv_shapes[-3]
        convt3_1 = self.conv2d_transpose(input=convt2_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 16, 32], \
            dilations=[1, 1, 1, 1], activation="elu", name="dec3_1")
        convt3_2 = self.conv2d(input=convt3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="dec3_2")

        return convt3_2

    def hypotheses(self, input, ksize=3, h_num=0, expand=True):

        print("\nHypotheses %d" %(h_num))
        convh1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 1], activation="sigmoid", name="hypo1_1-%d" %(h_num))

        if(expand): convh1_1 = tf.expand_dims(input=convh1_1, axis=0)

        return convh1_1

    def discriminator(self, input, ksize=3):

        print("\nDiscriminate-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="elu", name="dis1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="dis1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")

        print("Discriminate-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="dis2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="dis2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")

        print("Discriminate-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="dis3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="dis3_2")

        print("Dense (Fully-Connected)")
        [n, h, w, c] = conv3_2.shape
        fulcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="disfc_1")
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="elu", name="disfc_2")
        disc_score = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=1, activation="sigmoid", name="disfc_3")
        disc_score = tf.compat.v1.clip_by_value(disc_score, 0+(1e-12), 1-(1e-12))

        return disc_score

    def split_z(self, z):

        z_mu = z[:, :self.z_dim]
        # z_mu = tf.compat.v1.clip_by_value(z_mu, -3+(1e-12), 3-(1e-12))
        z_sigma = z[:, self.z_dim:]
        z_sigma = tf.compat.v1.clip_by_value(z_sigma, 1e-12, 1-(1e-12))

        return z_mu, z_sigma

    def sample_z(self, mu, sigma):

        epsilon = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        sample = mu + (sigma * epsilon)

        return sample

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_inputs, num_outputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_outputs, num_inputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-2]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
