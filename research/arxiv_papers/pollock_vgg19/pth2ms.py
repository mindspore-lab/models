import sys
from pathlib import Path

import numpy as np
import torch
import mindspore as ms
from mindspore.train.serialization import save_checkpoint


def pytorch2mindspore(pth_path, ckpt_path=None, prefix=None):

	if not ckpt_path:
		suffix = ''
		if pth_path.endswith('.pth'):
			suffix = '.pth'
		elif pth_path.endswith('.pth.tar'):
			suffix = '.pth.tar'
		else:
			raise Exception()
		ckpt_path = pth_path.replace(suffix, '.ckpt')

	para_dict = torch.load(pth_path)
	if 'state_dict' in para_dict:
		para_dict = para_dict['state_dict']

	new_params_list = []
	for i, (name, parameter) in enumerate(para_dict.items()):
		new_name = f'{prefix}.{name}' if prefix else name
		print(f'para{i}:  {name} ==> {new_name}')
		param_dict = dict(name=new_name, data=ms.Tensor(parameter.numpy()))
		new_params_list.append(param_dict)

	ms.save_checkpoint(new_params_list, ckpt_path)

	print('convert done.')
	print(f'mindspore model: {ckpt_path}')



def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_pth', type=str, required=True, 
		help='path to PyTorch pretrained model file (.pth)')
	parser.add_argument('-m', '--model_name', required=True, 
		choices=['vgg', 'decoder', 'SCT'])
	parser.add_argument('-o', '--output_ckpt', type=str, default=None, 
		help='path to save mindspore model file (.ckpt)')
	args = parser.parse_args()

	assert Path(args.input_pth).is_file()
	prefix = None if args.model_name == 'vgg' else args.model_name
	pytorch2mindspore(args.input_pth, ckpt_path=args.output_ckpt, prefix=prefix)


if __name__ == "__main__":
	main()
