# function： convert raw mindir model file to hardware-optimized mindir model file

import os

import mindspore_lite as mslite
import argparse


if __name__ == "__main__":
    sizes = ['384_640', '512_512', '512_640', '640_384', '640_512']

    config_file = './convert/config.cni'
    converter = mslite.Converter()
    converter.optimize = "ascend_oriented"

    for size in sizes:
        input_file = f'./models/wukong_youhua_{size}_graph.mindir'  # raw mindir file path
        output_file= f'./models/wukong_youhua_{size}_out'  # output file path, suffix '_graph.mindir' will be automaticly add
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        converter.convert(fmk_type=mslite.FmkType.MINDIR, model_file=input_file,
                          output_file=output_file, config_file=config_file)
