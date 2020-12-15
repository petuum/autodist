import glob
import os
import shutil

data_dir = '/home/christy.li/autosync_dataset_release'
for cluster in ['cluster1', 'cluster2']:
	folders = glob.glob(os.path.join(data_dir, cluster, '*'))
	print('\n{} folders in {}'.format(len(folders), cluster))
	for folder in folders:
		print('\nfolder', folder)
		resource_folder = os.path.join(folder, 'resource_specs')
		if os.path.exists(resource_folder):
			resource_files = glob.glob(os.path.join(resource_folder, '*'))
			shutil.copy(resource_files[0], folder)
			os.rename('/'.join(resource_files[0].split('/')[:-2]+[resource_files[0].split('/')[-1]]),
			          os.path.join(folder, 'resource_spec.yml'))
			shutil.rmtree(resource_folder)

