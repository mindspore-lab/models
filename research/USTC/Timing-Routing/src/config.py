class config():
    def __init__(self):
        self.seed = 1
        self.device_target = "None"
        self.context_mode = "pynative"  # should be in ['graph', 'pynative']
        self.device_num = 1
        self.device_id = 0

if __name__ == '__main__':
    cfg = config()
    print(cfg.seed)
    