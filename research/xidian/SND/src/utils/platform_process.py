import subprocess
import os
import time


def cmd_shell(cmd, shell=True):
    print('exec {}'.format(cmd))
    p = subprocess.Popen(cmd, shell=shell,
                         # stdout=subprocess.PIPE,  # -1 标准输出（演示器、终端) 保存到管道中以便进行操作
                         # stderr=subprocess.PIPE,  # 标准错误，保存到管道
                         )
    p.wait()


def platform_preprocess(args):
    assert not args.model_arts or not args.openi, 'Only one platform can be selected'

    if not args.model_arts and not args.openi:
        print('Not use any platform')
        return False

    if args.device_id % 8 != 0 and args.device_num > 1:
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)

    platform = None
    platform = 'ModelArts' if args.model_arts else platform
    platform = 'Openi-{}'.format(args.platform) if args.openi else platform

    local_data_dir = '/cache/data'
    local_ckpt_dir = '/cache/pretrained'
    local_output_dir = '/cache/output'

    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)

    cmd_shell("ln -s {} {}".format(local_data_dir, './data'), shell=True)
    cmd_shell('ln -s {} {}'.format(local_ckpt_dir, './pretrained'), shell=True)
    cmd_shell('ln -s {} {}'.format(local_output_dir, './output'), shell=True)
    args.output_dir = './output'

    if args.model_arts:
        import moxing
        remote_data_url = args.data_url
        remote_ckpt_url = args.ckpt_url
        moxing.file.copy_parallel(src_url=remote_data_url, dst_url=local_data_dir)
        moxing.file.copy_parallel(src_url=remote_ckpt_url, dst_url=local_ckpt_dir)

    if args.openi:
        import moxing
        assert args.platform in ["QZ", "ZS"]

        if args.platform == 'ZS':
            from src.utils.openi import c2net_multidataset_to_env as DatasetToEnv
            from src.utils.openi import pretrain_to_env
            ###拷贝云上的数据集到训练镜像
            DatasetToEnv(args.multi_data_url, local_data_dir)
            ###拷贝云上的预训练模型文件到训练环境
            pretrain_to_env(args.pretrain_url, local_ckpt_dir)

        if args.platform == 'QZ':
            from src.utils.openi import openi_multidataset_to_env as DatasetToEnv
            from src.utils.openi import pretrain_to_env
            ###拷贝云上的数据集到训练镜像
            DatasetToEnv(args.multi_data_url, local_data_dir)
            ###拷贝云上的预训练模型文件到训练环境
            pretrain_to_env(args.pretrain_url, local_ckpt_dir)
    f = open("/cache/download_input.txt", 'w')
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")

    print('Use platform : {} , preprocess Over!'.format(platform))
    return True


def platform_postprocess(args):
    assert not args.model_arts or not args.openi, 'Only one platform can be selected'

    if not args.model_arts and not args.openi:
        print('Not use any platform')
        return False

    platform = None
    platform = 'ModelArts' if args.model_arts else platform
    platform = 'Openi-{}'.format(args.platform) if args.openi else platform

    local_output_dir = './output'

    if args.model_arts:
        import moxing
        moxing.file.copy_parallel(src_url=local_output_dir, dst_url=args.train_url)

    if args.openi:
        import moxing
        assert args.platform in ["QZ", "ZS"]

        if args.platform == 'ZS':
            from src.utils.openi import env_to_openi
            ###上传训练结果到启智平台
            env_to_openi(local_output_dir, args.model_url)

        if args.platform == 'QZ':
            from src.utils.openi import env_to_openi
            ###上传训练结果到启智平台
            env_to_openi(local_output_dir, args.train_url)
    print('Use platform : {} , postprocess Over!'.format(platform))
    return True
