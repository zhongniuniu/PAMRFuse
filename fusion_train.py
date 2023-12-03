# Train the DenseFuse Net
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import scipy.io as scio
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import time
import scipy.ndimage

from fusion_model.nolocal_Generator import Generator
from fusion_model.Discriminator import Discriminator
import tensorlayer as tl
from fusion_model.LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from datetime import datetime

patch_size = 296
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

LEARNING_RATE = 0.0002
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8

retrain = False
model_path ='./models/'


def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE):
	tf.disable_v2_behavior()
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# create the graph
	with tf.Graph().as_default(), tf.Session(config=config) as sess:
		SOURCE_oe = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'OE_IMG')  #MRI
		SOURCE_ue = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'UE_IMG')  #PAT
		#GT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 3), name = 'GT')
		print('source img shape:', SOURCE_oe.shape) # (24, 144, 144, 1)

		# upsampling vis and ir images
		G = Generator('Generator')
		generated_img = G.transform(oe_img = SOURCE_oe, ue_img = SOURCE_ue, is_training=True)
		print('generate img shape:', generated_img.shape)


		D = Discriminator('Discriminator')
		D_real = D.discrim(SOURCE_oe, reuse = False)
		D_fake = D.discrim(generated_img, reuse = True)

		D1 = Discriminator('Discriminator1')
		D_real1 = D1.discrim(SOURCE_ue, reuse = False)
		D_fake1 = D1.discrim(generated_img, reuse = True)

		#######  LOSS FUNCTION
		# Loss for Generator
		G_loss_adv = -tf.reduce_mean(tf.log(D_fake + eps)) -  tf.reduce_mean(tf.log(D_fake1 + eps))

		grad_oe = grad(SOURCE_oe)
		grad_ue = grad(SOURCE_ue)
		# grad_fuse = grad(generated_img)
		# gradient_loss = L1_LOSS(grad_oe-grad_fuse) + L1_LOSS(grad_ue-grad_fuse)
		# G_loss_content =  0.6 * gradient_loss
		LOSS_IR = 0.6 * Fro_LOSS(generated_img - SOURCE_ue) + Fro_LOSS(generated_img - SOURCE_oe)#比较真实的矩阵和估计的矩阵值之间的误差1.2/1
		LOSS_VIS = L1_LOSS(grad(generated_img) - grad_oe) + L1_LOSS(grad(generated_img) - grad_ue)#模型预测值f(x)和真实值y之间绝对差值的平均值
		G_loss_norm = LOSS_IR + LOSS_VIS
		G_loss = G_loss_adv +  G_loss_norm


		# Loss for Discriminator
		D_loss_real = -tf.reduce_mean(tf.log(D_real + eps))
		D_loss_fake = -tf.reduce_mean(tf.log(1. - D_fake + eps))
		D_loss = D_loss_fake + D_loss_real

		D_loss_real1 = -tf.reduce_mean(tf.log(D_real1 + eps))
		D_loss_fake1 = -tf.reduce_mean(tf.log(1. - D_fake1 + eps))
		D1_loss = D_loss_fake1 + D_loss_real1

		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')
		theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1')

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		with tf.control_dependencies(update_ops):
			# G_solver_adv = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_adv, global_step = current_iter, var_list = theta_G)
			# G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter, var_list = theta_G)
			# D_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D_loss, global_step = current_iter, var_list = theta_D)

			G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_adv, global_step=current_iter,var_list=theta_G)
			G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0, beta2=0.9).minimize(G_loss, global_step = current_iter, var_list = theta_G)
			D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * 2).minimize(D_loss, global_step = current_iter, var_list = theta_D)
			D1_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * 2).minimize(D1_loss,global_step=current_iter,var_list=theta_D1)

		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D]
		clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]

		sess.run(tf.global_variables_initializer())
		g_list = tf.global_variables()
		saver = tf.train.Saver(var_list = g_list, max_to_keep = 200)

		tf.summary.scalar('G_Loss_content', G_loss)
		tf.summary.scalar('G_Loss_adv', G_loss_adv)
		tf.summary.scalar('D_Loss', D_loss)
		tf.summary.scalar('D_real', tf.reduce_mean(D_real))
		tf.summary.scalar('D_fake', tf.reduce_mean(D_fake))
		tf.summary.scalar('Learning_rate', learning_rate)

		tf.summary.image('oe', SOURCE_oe, max_outputs = 3)
		tf.summary.image('ue', SOURCE_ue, max_outputs = 3)
		tf.summary.image('fused_result', generated_img, max_outputs = 3)
		# tf.summary.image('groundtruth', GT, max_outputs = 3)
		# tf.summary.image('attention_map', tf.expand_dims(attention_map, axis=-1), max_outputs = 3)

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		# if retrain:
		# 	model_save_path = model_path + str(model_num) + '/' + str(model_num) + '.ckpt'
		# 	print("retrain: model:", model_save_path)
		# 	saver.restore(sess, model_save_path)
		# 	step = model_num
		# else:
		step = 0


		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)

			for batch in range(n_batches):
				step += 1
				current_iter = step
				oe_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0] #MRI
				ue_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1] #PAT
				oe_batch = np.expand_dims(oe_batch, -1)
				ue_batch = np.expand_dims(ue_batch, -1)
				#gt_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 6:9]

				FEED_DICT = {SOURCE_oe: oe_batch, SOURCE_ue: ue_batch}

				it_d = 0
				it_d1 = 0
				it_g = 0
				# run the training step
				if batch % 2 == 0:
					sess.run([D_solver, clip_D], feed_dict = FEED_DICT)
					it_d += 1
					sess.run([D1_solver, clip_D1], feed_dict=FEED_DICT)
					it_d1 += 1
				else:
					sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
					it_g += 1
				g_loss, d_loss, d1_loss = sess.run([G_loss, D_loss, D1_loss], feed_dict=FEED_DICT)

				# if batch % 2 == 0:
				# 	#d_fake, d_fake1, d_real, d_real1 = sess.run([tf.reduce_mean(D_fake), tf.reduce_mean(D_fake1), tf.reduce_mean(D_real), tf.reduce_mean(D_real1)], feed_dict = FEED_DICT)
				# 	while d_loss > 1.38 and (it_d < 20):
				# 		sess.run([D_solver, clip_D], feed_dict = FEED_DICT)
				# 		d_loss = sess.run(D_loss, feed_dict = FEED_DICT)
				# 		it_d += 1
				# 	while d1_loss > 1.38 and (it_d1 < 20):
				# 		sess.run([D1_solver, clip_D1], feed_dict=FEED_DICT)
				# 		d1_loss= sess.run(D1_loss, feed_dict=FEED_DICT)
				# 		it_d1 += 1
				# else:
				# 	while (d_loss < 1.37 or d1_loss < 1.37) and it_g < 20:
				# 		sess.run([G_GAN_solver, clip_G], feed_dict=FEED_DICT)
				# 		g_loss, d_loss, d1_loss = sess.run([G_loss, D_loss, D1_loss], feed_dict=FEED_DICT)
				# 		it_g += 1
				# 	while (g_loss > 140000) and it_g < 20:
				# 		sess.run([G_solver, clip_G], feed_dict=FEED_DICT)
				# 		g_loss = sess.run(G_loss, feed_dict=FEED_DICT)
				# 		it_g += 1

				if batch % 10 == 0:
					lr = sess.run(learning_rate)
				# print('batch:%s, it_g:%s, it_d:%s' % ((step % n_batches), it_g, it_d))

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)

				# if batch % 20 == 0:
				# 	elapsed_time = datetime.now() - start_time
				# 	g_loss_content = sess.run(G_loss_content, feed_dict = FEED_DICT)
				# 	g_loss_adv, d_loss, d1_loss = sess.run([G_loss_adv, D_loss, D1_loss], feed_dict = FEED_DICT)
				# 	d_fake, d_real = sess.run([tf.reduce_mean(D_fake), tf.reduce_mean(D_real)], feed_dict = FEED_DICT)
				# 	d_fake1, d_real1 = sess.run([tf.reduce_mean(D_fake1), tf.reduce_mean(D_real1)], feed_dict=FEED_DICT)
				# 	gra_loss=sess.run([tf.reduce_mean(gradient_loss)], feed_dict=FEED_DICT)

				print("Epoch:%s, batch: %s/%s, step: %s" % (epoch + 1, (step % n_batches), n_batches, step))
					# print('G_loss_content:%s' % (g_loss_content))
					# print('gradient_loss:%s' % (gra_loss))
					# print('G_loss_adv:%s, D_loss:%s' % (g_loss_adv, d_loss))
				print('G_loss:%s' % (g_loss))
				print('D_loss:%s, D_loss1:%s' % (d_loss, d1_loss))
				plot_G_loss = []
				plot_G_loss.append(g_loss)

					# print('D_fake:%s, D_real:%s' % (d_fake, d_real))
					# print('D_fake1:%s, D_real1:%s' % (d_fake1, d_real1))
					# print("elapsed_time:%s\n" % (elapsed_time))


				if (step % 10 == 0) or (step % n_batches == 0):
					print("save path:", save_path)
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')
			# np.save('/mnt/dfc_data1/home/linyusen/zhongyutian/RSAF-GAN/fusion_result/loss/Gloss_epoch_{}'.format(epoch),plot_G_loss)
		writer.close()
		saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g


def rgb2ihs(rgbimg):
	r = rgbimg[:, :, 0]
	g = rgbimg[:, :, 1]
	b = rgbimg[:, :, 2]
	i = tf.expand_dims(1 / np.sqrt(3) * r + 1 / np.sqrt(3) * g + 1 / np.sqrt(3) * b, -1)
	h = tf.expand_dims(1 / np.sqrt(6) * r + 1 / np.sqrt(6) * g - 2 / np.sqrt(6) * b, -1)
	v = tf.expand_dims(1 / np.sqrt(2) * r - 1 / np.sqrt(2) * g, -1)
	ihsimg = tf.concat([i, h, v], -1)
	return ihsimg


def L1_LOSS(batchimg):
	L1_norm = tf.reduce_sum(tf.abs(batchimg), axis = [1, 2])/(patch_size * patch_size)
	E = tf.reduce_mean(L1_norm)
	return E
