import os
import warnings
from glob import glob
import numpy as np
import cv2


class Captcha(object):
	def __init__(self, train_path: str = './sampleCaptchas', threshold: int = 40):
		'''
		Load the training captcha images and generate feature pattern of each character.
		args:
			train_path: path to sample captcha images for training
			threshold: threshold for binary map generation
		'''

		# Search images in train_path.
		self.train_im_list = []
		self.train_txt_list = []
		self.label_txt_list = []

		self.val_im_list = []

		for im_path in sorted(glob(train_path + '/input/*.jpg')):
			input_txt = im_path.replace('.jpg', '.txt')
			output_txt = input_txt.replace('input', 'output')
			if os.path.exists(input_txt):
				if os.path.exists(output_txt):
					self.train_im_list.append(im_path)
					self.train_txt_list.append(input_txt)
					self.label_txt_list.append(output_txt)
				else:
					warnings.warn(f'Could not find file "{output_txt}". Removed the image from training list.', UserWarning)
					self.val_im_list.append(im_path)
			else:
				warnings.warn(f'Could not find file "{input_txt}". Removed the image from training list.', UserWarning)
				self.val_im_list.append(im_path)

		self.threshold = threshold
		self.label2id = {}
		self.id2sample = []

		skipped = []
		for i, im_path in enumerate(self.train_im_list):
			input_txt = self.train_txt_list[i]
			output_txt = self.label_txt_list[i]
			im = cv2.imread(im_path)
			h, w, c = im.shape

			with open(input_txt, 'r') as f:
				input_meta = f.readlines()
			with open(output_txt, 'r') as f:
				label = f.read().strip()
			assert len(label) == 5

			# Check if image's height and width match the values in its corresponding intput text file.
			h_meta, w_meta = input_meta[0].strip().split(' ')
			if not int(h_meta) == h or not int(w_meta) == w:
				warnings.warn(f'''
					Found resolution mismatch: "{im_path}": Meta data: ({h_meta}, {w_meta}), image resolution: ({h}, {w}). Removed the image from training list.
				''', UserWarning)
				skipped.append(i)
				continue

			# Confirm each pixel has identical values across three channels and matches the values in its corresponding intput text file.
			for i_h, line_h in enumerate(input_meta[1:]):
				for i_w, line_w in enumerate(line_h.strip().split(' ')):
					b, g, r = im[i_h, i_w, :]
					b_meta, g_meta, r_meta = line_w.strip().split(',')
					assert b == g == r
					assert int(b_meta) == b and int(g_meta) == g and int(r_meta) == r

			# Convert image into binary map.
			binary = (im[:, :, 0] < self.threshold).astype(int)

			# Remove blank areas by checking the sum of its rows or columns equals zero.
			start = 0
			end = 0
			row_sum = np.sum(binary, axis=1)
			for i_r, row in enumerate(row_sum):
				if row:
					if start == 0:
						start = i_r
					else:
						end = i_r
			end += 1

			binary = binary[start:end, :]
			segments = []
			start = 0
			end = 0
			col_sum = np.sum(binary, axis=0)
			for i_c, col in enumerate(col_sum):
				if col:
					if start == 0:
						start = i_c
					else:
						end = i_c
				elif end != 0:
					segments.append([start, end+1])
					start = 0
					end = 0

			# Generate dictionary to map the character segments to its corresponding characters.
			for i_char, seg in enumerate(segments):
				if not label[i_char] in self.label2id:
					self.label2id[label[i_char]] = len(self.id2sample)
					self.id2sample.append(binary[:, seg[0]:seg[1]])
				else:
					# Check if the same character segment set is used for all images across the dataset.
					assert seg[1] - seg[0] == self.id2sample[self.label2id[label[i_char]]].shape[1]
					assert np.all(self.id2sample[self.label2id[label[i_char]]] == binary[:, seg[0]:seg[1]])

		# Map the characters to its corresponding patterns.
		self.id2label = {}
		for letter in self.label2id:
			self.id2label[self.label2id[letter]] = letter

		# Images without input or output text file or having mismatches will be moved into validation dataset.
		num_im = len(self.train_im_list)
		self.val_im_list += [self.train_im_list[i] for i in skipped]
		self.train_im_list = [self.train_im_list[i] for i in range(num_im) if i not in skipped]
		self.train_txt_list = [self.train_txt_list[i] for i in range(num_im) if i not in skipped]
		self.label_txt_list = [self.label_txt_list[i] for i in range(num_im) if i not in skipped]
		print(f'{len(self.train_im_list)} training images enrolled successfully. Skipped images: {self.val_im_list}.')


	def __call__(self, im_path, save_path: str = 'output'):
		'''
		Algo for identify the given captcha image.
		args:
			im_path: .jpg image path to load and to infer
			save_path: output file path to save the one-line outcome
		'''
		answer = []
		im = cv2.imread(im_path)

		# Convert image into binary map.
		binary = (im[:, :, 0] < self.threshold).astype(int)

		# Remove blank areas by checking the sum of its rows or columns equals zero.
		start = 0
		end = 0
		row_sum = np.sum(binary, axis=1)
		for i_r, row in enumerate(row_sum):
			if row:
				if start == 0:
					start = i_r
				else:
					end = i_r
		end += 1

		binary = binary[start:end, :]
		segments = []
		start = 0
		end = 0
		col_sum = np.sum(binary, axis=0)
		for i_c, col in enumerate(col_sum):
			if col:
				if start == 0:
					start = i_c
				else:
					end = i_c
			elif end != 0:
				segments.append([start, end+1])
				start = 0
				end = 0

		# Compare the characters with the patterns in saved character dictionary.
		for i_char, seg in enumerate(segments):
			for i_sample, sample in enumerate(self.id2sample):
				if seg[1] - seg[0] != sample.shape[1]:
					continue
				if np.all(binary[:, seg[0]:seg[1]] == sample):
					answer.append(self.id2label[i_sample])

		# Combine the output and write into file.
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		file_name = os.path.basename(im_path)
		if 'input' in file_name:
			output_path = f"{save_path}/{file_name.replace('input', 'output').split('.')[0]}.txt"
		else:
			output_path =  f"output/{file_name.split('.')[0]}.txt"
		with open(output_path, 'w') as f:
			answer.append('\n')
			f.write(''.join(answer))


if __name__ == '__main__':
	# For validation
	captcha_identifier = Captcha()
	for im_path in captcha_identifier.val_im_list:
		captcha_identifier(im_path)
	for im_path in captcha_identifier.train_im_list:
		captcha_identifier(im_path)
	print('Captcha identification Finished. Outputs are saved in ./output')