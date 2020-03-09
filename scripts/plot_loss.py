import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage

def read_loss(file):
	with open(file, 'r') as f:
		# print(f.readline())
		# a=f.readline()
		# print(a)
		inp = f.read()


	patterns = {
		'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \|( S: (?P<s>\S+) \|)? T: (?P<t>\S+)'),
		'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
	}
	data = {key: [] for key in patterns}
	val_box=[]
	val_mask=[]
	import ast
	for line in inp.split('\n'):
		# print(line)
		try:
			dict = ast.literal_eval(line)
			# print(dict)
			# if dict['type']=='train':
			# 	print('t')
			if dict['type']=='val':
				val_box.append(dict['data']['box']['all'])
				val_mask.append(dict['data']['mask']['all'])
				data['train'].append(dict['data']['mask']['all'])
		except:
			print('a')
		# for key, pattern in patterns.items():
		# 	f = pattern.search(line)
		#
		# 	if f is not None:
		# 		datum = f.groupdict()
		# 		for k, v in datum.items():
		# 			if v is not None:
		# 				try:
		# 					v = float(v)
		# 				except ValueError:
		# 					pass
		# 				datum[k] = v
		#
		# 		if key == 'val':
		# 			datum = (datum, data['train'][-1])
		# 		data[key].append(datum)
		# 		break
	print(len(val_mask))
	return val_mask, val_box

def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	loss_names = ['BBox Loss', 'Conf Loss', 'Mask Loss']

	x = [x['iteration'] for x in data]
	plt.plot(x, smoother([y['b'] for y in data]))
	plt.plot(x, smoother([y['c'] for y in data]))
	plt.plot(x, smoother([y['m'] for y in data]))

	if data[0]['s'] is not None:
		plt.plot(x, smoother([y['s'] for y in data]))
		loss_names.append('Segmentation Loss')

	plt.legend(loss_names)
	plt.show()

def plot_val(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
	plt.xlabel('Epoch')
	plt.ylabel('mAP')

	# x = [x[1]['epoch'] for x in data if x[0]['type'] == 'box']
	# plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'box'])
	# plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'mask'])
	plt.plot(range(len(val_mask)), val_box)
	plt.plot(range(len(val_mask)), val_mask)

	plt.legend(['BBox mAP', 'Mask mAP'])
	plt.savefig('plot.png')

	plt.show()


if len(sys.argv) > 2 and sys.argv[2] == 'val':
	val_mask1, val_box1 = read_loss('/home/tomerweiss/yolact/logs/yolact_resnet50_my1.log')
	val_mask2, val_box2 = read_loss('/home/tomerweiss/yolact/logs/yolact_resnet50.log')
	plt.title(os.path.basename(sys.argv[1]) + 'Bounding Box - Validation')
	plt.xlabel('Epoch')
	plt.ylabel('mAP')

	# x = [x[1]['epoch'] for x in data if x[0]['type'] == 'box']
	# plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'box'])
	# plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'mask'])
	plt.plot(range(len(val_mask1)), val_box1)
	plt.plot(range(len(val_mask2)), val_box2)

	plt.legend(['With Boundary Loss', 'Original'])
	plt.savefig('box_plot.png')

	plt.show()
	# plot_val(data['val'])
else:
	plot_train(data['train'])
