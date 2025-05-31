from loss import tversky_kahneman_loss
from model.encoderModel import EncoderModel
from model.fullModel import FullModel
import torch
import torch.nn as nn
import gc
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random


# args
dataset = 'ISIC2016'
batch_size = 16
epochs = 100
in_channels = 3
training_strategy = 1
saving_each_epoch = True

nums_class = 1   # for binary classification
nums_mask = 1    # for segment skin lesion


# load data and config

skin_images = np.load('loaded_data/' + dataset + '/train/skin_images.npy')
mask_labels = np.load('loaded_data/' + dataset + '/train/mask_labels.npy')
class_labels = np.load('loaded_data/' + dataset + '/train/class_labels.npy')
mask_labels = mask_labels.astype("uint16")


index = np.arange(len(skin_images))
num_batches = int(math.floor(index.shape[0]/batch_size))
filter_list = [2*(2**i) for i in range(5)]
out_dict = {'class':nums_class, 'image':nums_mask}


if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = 'cpu'


class_critiation = nn.BCELoss()
mask_critiation = tversky_kahneman_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

seq = iaa.Sequential([
    iaa.Crop(px=(0, 10)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.LinearContrast((0.9, 1.1)),
    iaa.Affine(translate_percent=0.08, scale=(0.85, 1.15), rotate=(-45, 45), cval=1),
])


if training_strategy in [1, 2, 3]:
	model = FullModel(in_channels, filter_list, out_dict).to(device)
else:
	model = EncoderModel(in_channels, filter_list, out_dict).to(device)


### Training ###

# Traing mask for strategy 2 
if training_strategy == 2:
	for e in range(0, int(epochs/2)):
	  print('Epochs', e+1)
	  mask_epoch_loss = 0

	  np.random.shuffle(index)

	  for batch in tqdm(range(0, num_batches)):

	    if batch == num_batches - 1:
	      index_batch = index[batch_size*batch:]
	    else:
	      index_batch = index[batch_size*batch : batch_size*(batch+1)]

	    batch_size_current = len(index_batch)

	    skin_images_aug = []
	    mask_labels_aug = []

	    for k in index_batch:
	      segmap = SegmentationMapsOnImage(mask_labels[k], shape=skin_images[k].shape)
	      images_aug, segmaps_aug = seq(image=skin_images[k], segmentation_maps=segmap)

	      skin_images_aug.append(images_aug)
	      mask_labels_aug.append(segmaps_aug.get_arr())

	    skin_images_aug = np.array(skin_images_aug)
	    mask_labels_aug = np.array(mask_labels_aug)

	    gc.collect()

	    skin_batch = torch.from_numpy(skin_images_aug).permute(0, 3, 1, 2).to(device, dtype=torch.float)
	    mask_batch = torch.from_numpy(mask_labels_aug).unsqueeze(1).to(device, dtype=torch.float)

	    y_class, y_mask = model(skin_batch)
	    mask_loss = mask_critiation(y_mask, mask_batch)

	    mask_epoch_loss += float(mask_loss)*batch_size_current

	    mask_loss.backward()
	    optimizer.step()
	    optimizer.zero_grad()

	    del y_class
	    del y_mask
	    del mask_loss
	    gc.collect()

	    del skin_images_aug
	    del mask_labels_aug
	    gc.collect()

	  print(' Mask loss =', mask_epoch_loss/len(index))

	if saving_each_epoch:
		torch.save(model.state_dict(), 'weights/segment_' + str(e+1) + '.pkl')


if training_strategy == 1:
	for e in range(0, epochs):
	  print('Epochs', e+1)
	  class_epoch_loss = 0
	  mask_epoch_loss = 0
	  epoch_loss = 0

	  np.random.shuffle(index)

	  for batch in tqdm(range(0, num_batches)):

	    if batch == num_batches - 1:
	      index_batch = index[batch_size*batch:]
	    else:
	      index_batch = index[batch_size*batch : batch_size*(batch+1)]

	    batch_size_current = len(index_batch)

	    skin_images_aug = []
	    mask_labels_aug = []
	    class_labels_aug = []

	    for k in index_batch:
	      segmap = SegmentationMapsOnImage(mask_labels[k], shape=skin_images[k].shape)
	      images_aug, segmaps_aug = seq(image=skin_images[k], segmentation_maps=segmap)

	      skin_images_aug.append(images_aug)
	      mask_labels_aug.append(segmaps_aug.get_arr())
	      class_labels_aug.append(class_labels[k])

	    skin_images_aug = np.array(skin_images_aug)
	    mask_labels_aug = np.array(mask_labels_aug)
	    class_labels_aug = np.array(class_labels_aug)

	    gc.collect()

	    skin_batch = torch.from_numpy(skin_images_aug).permute(0, 3, 1, 2).to(device, dtype=torch.float)
	    mask_batch = torch.from_numpy(mask_labels_aug).unsqueeze(1).to(device, dtype=torch.float)
	    class_batch = torch.from_numpy(class_labels_aug).to(device, dtype=torch.float)


	    y_class, y_mask = model(skin_batch)
	    class_loss = class_critiation(y_class, class_batch)
	    mask_loss = mask_critiation(y_mask, mask_batch)
	    batch_loss = class_loss + mask_loss

	    class_epoch_loss += float(class_loss)*batch_size_current
	    mask_epoch_loss += float(mask_loss)*batch_size_current
	    epoch_loss += float(batch_loss)*batch_size_current

	    batch_loss.backward()
	    optimizer.step()
	    optimizer.zero_grad()

	    del y_class
	    del y_mask
	    del class_loss
	    del mask_loss
	    del batch_loss
	    gc.collect()

	    del skin_images_aug
	    del mask_labels_aug
	    del class_labels_aug
	    gc.collect()

	  print(' Class loss =', class_epoch_loss/len(index))
	  print(' Mask loss =', mask_epoch_loss/len(index))
	  print(' Total loss =', epoch_loss/len(index))

	  if saving_each_epoch:
	  	torch.save(model.state_dict(), 'weights/strategy1_' + str(e+1) + '.pkl')


if training_strategy in [2, 3]:
	for e in range(0, epochs):
	  print('Epochs', e+1)
	  class_epoch_loss = 0
	  mask_epoch_loss = 0
	  epoch_loss = 0

	  model.train()
	  np.random.shuffle(index)

	  for batch in tqdm(range(0, num_batches)):

	    if batch == num_batches - 1:
	      index_batch = index[batch_size*batch:]
	    else:
	      index_batch = index[batch_size*batch : batch_size*(batch+1)]

	    batch_size_current = len(index_batch)

	    skin_images_aug = []
	    mask_labels_aug = []
	    class_labels_aug = []

	    for k in index_batch:
	      segmap = SegmentationMapsOnImage(mask_labels[k], shape=skin_images[k].shape)
	      images_aug, segmaps_aug = seq(image=skin_images[k], segmentation_maps=segmap)

	      skin_images_aug.append(images_aug)
	      mask_labels_aug.append(segmaps_aug.get_arr())
	      class_labels_aug.append(class_labels[k])

	    skin_images_aug = np.array(skin_images_aug)
	    mask_labels_aug = np.array(mask_labels_aug)
	    class_labels_aug = np.array(class_labels_aug)

	    gc.collect()

	    skin_batch = torch.from_numpy(skin_images_aug).permute(0, 3, 1, 2).to(device, dtype=torch.float)
	    mask_batch = torch.from_numpy(mask_labels_aug).unsqueeze(1).to(device, dtype=torch.float)
	    class_batch = torch.from_numpy(class_labels_aug).to(device, dtype=torch.float)


	    y_class, y_mask = model(skin_batch)
	    class_loss = class_critiation(y_class, class_batch)

	    epoch_loss += float(class_loss)*batch_size_current

	    class_loss.backward()
	    optimizer.step()
	    optimizer.zero_grad()

	    del y_class
	    del y_mask
	    del class_loss
	    gc.collect()

	    del skin_images_aug
	    del mask_labels_aug
	    del class_labels_aug
	    gc.collect()

	  print(' Class loss =', epoch_loss/len(index))
	  if saving_each_epoch:
	  	torch.save(model.state_dict(), 'weights/strategy' + int(training_strategy) + '_' + str(e+1) + '.pkl')


if training_strategy == 4:
	for e in range(0, epochs):
	  print('Epochs', e+1)
	  class_epoch_loss = 0
	  mask_epoch_loss = 0
	  epoch_loss = 0

	  np.random.shuffle(index)

	  for batch in tqdm(range(0, num_batches)):

	    if batch == num_batches - 1:
	      index_batch = index[batch_size*batch:]
	    else:
	      index_batch = index[batch_size*batch : batch_size*(batch+1)]

	    batch_size_current = len(index_batch)

	    skin_images_aug = []
	    mask_labels_aug = []
	    class_labels_aug = []

	    for k in index_batch:
	      segmap = SegmentationMapsOnImage(mask_labels[k], shape=skin_images[k].shape)
	      images_aug, segmaps_aug = seq(image=skin_images[k], segmentation_maps=segmap)

	      skin_images_aug.append(images_aug)
	      mask_labels_aug.append(segmaps_aug.get_arr())
	      class_labels_aug.append(class_labels[k])

	    skin_images_aug = np.array(skin_images_aug)
	    mask_labels_aug = np.array(mask_labels_aug)
	    class_labels_aug = np.array(class_labels_aug)

	    gc.collect()

	    skin_batch = torch.from_numpy(skin_images_aug).permute(0, 3, 1, 2).to(device, dtype=torch.float)
	    mask_batch = torch.from_numpy(mask_labels_aug).unsqueeze(1).to(device, dtype=torch.float)
	    class_batch = torch.from_numpy(class_labels_aug).to(device, dtype=torch.float)


	    y_class = model(skin_batch)
	    class_loss = class_critiation(y_class, class_batch)

	    class_epoch_loss += float(class_loss)*batch_size_current

	    class_loss.backward()
	    optimizer.step()
	    optimizer.zero_grad()

	    del y_class
	    del class_loss
	    gc.collect()

	    del skin_images_aug
	    del mask_labels_aug
	    del class_labels_aug
	    gc.collect()

	  print(' Class loss =', class_epoch_loss/len(index))

	  if saving_each_epoch:
	  	torch.save(model.state_dict(), 'weights/strategy4_' + str(e+1) + '.pkl')

	  	