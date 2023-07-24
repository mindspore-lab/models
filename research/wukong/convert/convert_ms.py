import os

import mindspore_lite as mslite
import argparse


if __name__ == "__main__":
    sizes = ['384_640', '512_512', '512_640', '640_384', '640_512']

    config_file = './convert/config.cni'
    converter = mslite.Converter()
    converter.optimize = "ascend_oriented"

    sizes = sizes[0:4]
    for size in sizes:
        # rename
        # os.rename(f'./models/0722_out/out_wukong_youhua_{size}_graph.mindir' ,f'./models/0722_out/wukong_youhua_{size}_out_graph.mindir')
        # os.rename(f'./models/0722_out/out_wukong_youhua_{size}.om' ,f'./models/0722_out/wukong_youhua_{size}_out.om')
        # os.rename(f'./models/0722_out/out_wukong_youhua_{size}_variables' ,f'./models/0722_out/wukong_youhua_{size}_out_variables')

        model_file = f'./models/wukong_youhua_{size}_graph.mindir'
        output_file= f'./models/0722_out/wukong_youhua_{size}_out'
        converter.convert(fmk_type=mslite.FmkType.MINDIR, model_file=model_file,
                          output_file=output_file, config_file=config_file)


