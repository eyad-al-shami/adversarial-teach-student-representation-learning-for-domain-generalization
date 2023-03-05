from glob import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


'''
The data files must have this structure

pacs:
  images
  splits

'''

class PACS_DATASET(Dataset):
	def __init__(self, domains = [], images_root_dir = "", transform=None, target_transform=None, verbose=False):
		assert domains != [], "source domains must be provided"
		assert not not images_root_dir, "root directory images must be provided"

		# check if the images_root_dir exists by throwing an error if it does not
		if not os.path.exists(images_root_dir):
			raise Exception(f"the directory {images_root_dir} does not exist")

		all_domains = ['cartoon', 'art_painting', 'photo', 'sketch']
		self.domains_to_index = {domain:i for i, domain in enumerate(all_domains)}
		self.index_to_domains = {i:domain for i, domain in enumerate(all_domains)}

		all_classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
		self.class_to_index = {class_:i for i, class_ in enumerate(all_classes)}
		self.index_to_class = {i:class_ for i, class_ in enumerate(all_classes)}

		data_triples = [] # image_path, domain_index, class_index
		for domain in domains:
			for class_ in all_classes:
				domain_dir = os.path.join(images_root_dir, "images", domain, class_)
				domain_class_images = glob(os.path.join(domain_dir, "*.png")) + glob(os.path.join(domain_dir, "*.jpg"))
				if (verbose):
					print(f"in the domain {domain} for the class {class_} there is {len(domain_class_images)}")
				data_triples.extend([(image_path, self.domains_to_index[domain], self.class_to_index[class_]) for image_path in domain_class_images])

		self.total_data = data_triples
		
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.total_data)

	def __getitem__(self, idx):
		img_path, domain, label = self.total_data[idx]
		image = Image.open(img_path)
		if self.transform:
			image = self.transform(image)
		return image, domain, label

def build_transforms(cfg, phase="train"):
    # training_tfms = transforms.Compose([
	#     transforms.CenterCrop(224),
	#     transforms.RandomHorizontalFlip(),
	#     # transforms.ColorJitter(brightness=.5, hue=.3),
	#     transforms.ToTensor(),
	#     # transforms.Normalize(mean=[0.8158, 0.7974, 0.7717], std=[0.2895, 0.3015, 0.3315]),
	# ])
	# basic_tfms = transforms.Compose([
	#     transforms.CenterCrop(224),
	#     transforms.ToTensor(),
	#     # transforms.Normalize(mean=[0.8158, 0.7974, 0.7717], std=[0.2895, 0.3015, 0.3315]),
	# ])
	tfms_list = []

	if phase == "train":
		pass
	elif phase == "val":
		pass
	elif phase == "test":
		pass
	elif phase == "warmup":
		tfms_list.append(transforms.RandomHorizontalFlip())
	
	tfms_list.append(transforms.ToTensor())

	if cfg.DATASET.NORMALIZE:
		tfms_list.append(transforms.Normalize(mean=cfg.DATASET.NORMALIZE.MEAN, std=cfg.DATASET.NORMALIZE.STD))
	
	
	tfms = transforms.Compose(tfms_list)
	return tfms

def get_dataset(cfg, phase="train", domains=[]):
	if domains == []:
		print("domains not provided, using the default training source domains.")
		domains = cfg.TRAIN.SOURCE_DOMAINS
	if cfg.DATASET.NAME == "PACS":
		tfms = build_transforms(cfg, phase)
		train_dataset = PACS_DATASET(domains=domains, images_root_dir=cfg.DATASET.ROOT, transform=tfms)
		train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS, persistent_workers=True)
	else:
		raise NotImplementedError
	return train_dataset, train_loader