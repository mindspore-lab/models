import json
import os

def parse_args(opts):
    savepath = os.path.join(opts.save_path, 'opts.json')
    if opts.resume or opts.test:
        with open(savepath, 'r', encoding='utf-8') as f:
            optDic = json.load(f)
        if opts.test:
            optDic['log_name'] = 'test' 
            optDic['test'] = True 
        if opts.resume:
            optDic['log_name'] = 'resume_from_S{}'.format(opts.resumeStep) 
            optDic['resume'] = True
            optDic['resumeStep'] = opts.resumeStep 
        
    else:
        assert os.path.exists(savepath) == False, '{} is already exists'.format(savepath)
        optDic = {}
        for k, v in sorted(vars(opts).items()):
            optDic[k] = v
        with open(savepath, 'w', encoding='utf-8') as f:
            json_dict = json.dump(optDic, indent=2, sort_keys=True, ensure_ascii=False, fp=f)
    return optDic

def print_options(opts, logger):
    message = '\n'
    message += '----------------- Options ---------------\n'
    for k, v in opts.items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    logger.info(message)
