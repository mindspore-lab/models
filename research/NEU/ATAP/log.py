import logging
import time

logger = logging.getLogger('demo')
logger.setLevel(level=logging.DEBUG)  # 相当于第一层过滤网

formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# 输出到文件

t = time.localtime()
cur_time = time.strftime("%y_%m_%d_%H_%M_%S", t)
file_handler = logging.FileHandler(str(cur_time) + 'train.log')  # 设置文件名，模式，编码
file_handler.setLevel(level=logging.INFO)  # 相当于第二层过滤网；第一层之后的内容再次过滤。
file_handler.setFormatter(formatter)

# 输出到控制台
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)  # 相当于第二层过滤网；第一层之后的内容再次过滤。
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)