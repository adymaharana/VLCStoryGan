#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate Video GAN

The difference of this GAN is replacing the original encoder using residual 2+1 encoder

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
import PIL
import functools

from .residual2p1 import R2Plus1D


def calculate_activation_statistics(vids, model, batch_size=32, dims=512,
									cuda=False, normalize=False, verbose=0, is_ref=False):
	"""Calculates the activations of the pool_3 layer for all images.

	Params:
		vids: image dataset
		model: Instance of inception model
		batch_size: Batch size of images for the model to process at once.
			Make sure that the number of samples is a multiple of the batch
			size, otherwise some samples are ignored. This behavior is retained
			to match the original FID score implementation.
		cuda: If set to True, use GPU
		normalize: If the value range of vids is [-1, 1], set to True to
			shift value range to [0, 1].
		verbose: If verbose > 0, show progressbar during evaluation
	Returns:
		mu: The mean over samples of the activations of the pool_3 layer of
			the inception model.
		sigma: The covariance matrix of the activations of the pool_3 layer of
			the inception model.
	"""
	model.eval()
	if cuda:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	model.to(device)

	with torch.no_grad():
		features = []
		dataloader = DataLoader(
			vids, batch_size=batch_size, num_workers= 10 if is_ref else 0,
			drop_last=True, shuffle=False)
		if verbose > 0:
			iter_dataset = tqdm(dataloader, dynamic_ncols=True)
		else:
			iter_dataset = dataloader
		for videos, _ in iter_dataset:
			# print(videos.shape)
			videos = videos.type(torch.FloatTensor).to(device)
			activation = model(videos)
			if activation.shape[2] != 1 or activation.shape[3] != 1:
				activation = F.adaptive_avg_pool2d(activation, output_size=(1, 1))
			features.append(activation.cpu().numpy().reshape(-1, dims))

		features = np.concatenate(features, axis=0)
		mu = np.mean(features, axis=0)
		sigma = np.cov(features, rowvar=False)

	return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

	Stable version by Dougal J. Sutherland.

	Params:
		mu1: Numpy array containing the activations of a layer of the
			inception net (like returned by the function 'get_predictions')
			for generated samples.
		mu2: The sample mean over activations, precalculated on an
			representative data set.
		sigma1: The covariance matrix over activations for generated samples.
		sigma2: The covariance matrix over activations, precalculated on an
			representative data set.

	Returns:
		The Frechet Distance.
	"""

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		print('fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	return (diff.dot(diff) +
			np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_score(r_imgs, g_imgs, batch_size=50, dims=512, cuda=False,
			  normalize=False, r_cache='reference_fid', verbose=0):
	model = R2Plus1D()
	# cache real dataset
	if r_cache and not r_cache.endswith('.npz'):
		r_cache = r_cache + '.npz'
	if r_cache and os.path.exists(r_cache):
		data = np.load(r_cache)
		m1, s1 = data['m1'], data['s1']
	else:
		m1, s1 = calculate_activation_statistics(r_imgs, model, batch_size,
												 dims, cuda, normalize, is_ref=True)
		if r_cache is not None:
			os.makedirs(os.path.dirname(r_cache), exist_ok=True)
			np.savez(r_cache, m1=m1, s1=s1)
	# compute generative image dataset
	m2, s2 = calculate_activation_statistics(g_imgs, model, batch_size, dims,
											 cuda, normalize, is_ref=False)
	fid_value = calculate_frechet_distance(m1, s1, m2, s2)

	return fid_value
