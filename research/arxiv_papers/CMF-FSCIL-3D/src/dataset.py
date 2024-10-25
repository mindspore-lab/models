from .common import *
import pickle
from .config.config import config
from .config.label import shapenet_label
from .utils import *
from data.shapenet.path import shapenet_train_path,shapenet_test_path
from data.shapenet.map import main_category_map,all_category_map
import torchvision.transforms as transforms
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from src.tokenizer import SimpleTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FileCheckError(Exception):
    pass

class ShapeNetTrain:

    def __init__(self, task_id, task_type="all"):
        path = shapenet_train_path
        self.task_id = task_id
        self.task_type = task_type
        self.main_pc_path = path.Main_PC_Path
        self.main_img_path = path.Main_IMG_Path
        self.main_text_path = path.Main_Text_Path
        self.all_img_path = path.All_IMG_Path
        self.all_pc_path = path.All_PC_Path
        self.all_text_path = path.All_Text_Path
        self.pc_all_path = path.PC_All
        self.index_file = path.Index_File
        self.npoints = config.npoints
        self.cache_dir = "data/shapenet/cache/train"
        self.check_dir = "log/check"
        self.main_category_map = all_category_map
        self.all_category_map = all_category_map
        self.increment_category_map = main_category_map

        
        
        self.tokenizer = SimpleTokenizer()

        self.permutation = np.arange(self.npoints)
        self.picked_rotation_degrees = list(range(0, 360, 36))
        self.picked_rotation_degrees = [
            (3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else
            str(degree) for degree in self.picked_rotation_degrees
        ]
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.prompt_dic = self.load_prompt()
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.check_dir,exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"task_{self.task_id}.pkl") if task_id != -2 else os.path.join(self.cache_dir, f"task_0.pkl")
        if self.task_id == -1:
            self.main_data_list, self.main_label_list, self.all_data_list, self.all_label_list = self.load_data_form_base()
        else:
            if os.path.exists(self.cache_file):
                print(f"Loading shapenet-train task {self.task_id} data from cache...")
                with open(self.cache_file, 'rb') as f:
                    if self.task_id == 0 or self.task_id == -2:
                        self.main_data_list, self.main_label_list, self.all_data_list, self.all_label_list = pickle.load(f)
                    else:
                        self.main_data_list, self.main_label_list = pickle.load(f)
            else:
                print(f"Loading shapenet-train task {self.task_id} data from original source...")
                if self.task_id == 0:
                    self.main_data_list, self.main_label_list, self.all_data_list, self.all_label_list = self.load_data_form_base()
                else:
                    self.main_data_list, self.main_label_list = self.load_data_for_increment()
                with open(self.cache_file, 'wb') as f:
                    if self.task_id == 0:
                        pickle.dump((self.main_data_list, self.main_label_list, self.all_data_list, self.all_label_list), f)
                    else:
                        pickle.dump((self.main_data_list, self.main_label_list), f)
        
        

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def process_pc(self, pc_path):
        
        data = np.load(pc_path)
        if len(data) > self.npoints:
            data = farthest_point_sample(data, self.npoints)
        data = self.pc_norm(data)
        data = random_point_dropout(data[None, ...])
        data = random_scale_point_cloud(data)
        data = shift_point_cloud(data)
        data = rotate_perturbation_point_cloud(data)
        data = rotate_point_cloud(data)
        data = data.squeeze()
        data = mindspore.Tensor(data).float()
        return data

    def process_img(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.train_transform(image)
        except:
            raise ValueError(f"Image is corrupted: {img_path}")
        return image

    def check_main_all(self, main_file_name, all_file_name):
        if os.path.splitext(main_file_name)[1] != os.path.splitext(all_file_name)[1]:
            raise FileCheckError(f"Extension mismatch: {main_file_name} vs {all_file_name}")

        main_base = os.path.splitext(os.path.basename(main_file_name))[0]
        all_base = os.path.splitext(os.path.basename(all_file_name))[0]

        if not all_base.startswith(main_base):
            raise FileCheckError(f"Base name mismatch: {main_file_name} vs {all_file_name}")

    def check_pc_img(self, pc_name, img_name):
        if not (pc_name.endswith('.npy') and img_name.endswith('.png')):
            raise FileCheckError(f"Extension mismatch: {pc_name} vs {img_name}")

        pc_base, pc_ext = os.path.splitext(os.path.basename(pc_name))
        img_base, img_ext = os.path.splitext(os.path.basename(img_name))

        if pc_ext != '.npy' or img_ext != '.png':
            raise FileCheckError(f"Incorrect file extensions: {pc_name}, {img_name}")

        pc_angle_match = re.match(r"(.*)_r(\d{3})$", pc_base)
        img_angle_match = re.match(r"(.*)_r_(\d{3})$", img_base)

        if not pc_angle_match:
            pc_base_name = pc_base
            img_base_name = img_base.split('_r_')[0]
            if pc_base_name != img_base_name:
                raise FileCheckError(f"Base name mismatch: {pc_name} vs {img_name}")

        else:
            pc_base_name = pc_angle_match.group(1)
            img_base_name = img_angle_match.group(1)

            if pc_base_name != img_base_name:
                raise FileCheckError(f"Base name mismatch: {pc_name} vs {img_name}")

            if pc_angle_match.group(2) != img_angle_match.group(2):
                raise FileCheckError(f"Angle mismatch: {pc_name} vs {img_name}")

    def process_category(self, cate):
        try:
            print(f"Processing category: {cate}")

            main_pc_files = sorted(f for f in os.listdir(os.path.join(self.main_pc_path, cate)) if f.endswith('.npy'))
            main_img_files = sorted(f for f in os.listdir(os.path.join(self.main_img_path, cate)) if f.endswith('_r_000.png'))

            assert len(main_pc_files) == len(main_img_files), f"Mismatch in number of files for category {cate}, for pc: {len(main_pc_files)}, for img: {len(main_img_files)}"

            all_pc_dir = os.path.join(self.all_pc_path, cate)
            sub_dirs = sorted(os.listdir(all_pc_dir))

            all_pc_files = []
            all_img_files = []

            for sub in sub_dirs:
                try:
                    sub_pc_dir = os.path.join(self.all_pc_path, cate, sub)
                    sub_img_dir = os.path.join(self.all_img_path, cate, sub)

                    all_pc_file = sorted(f for f in os.listdir(sub_pc_dir) if f.endswith('.npy'))
                    all_img_file = sorted(f for f in os.listdir(sub_img_dir) if f.endswith('.png'))

                    num_groups = len(all_pc_file) // 10

                    selected_pc_files = []
                    selected_img_files = []

                    for i in range(num_groups):
                        angle = str(random.choice(self.picked_rotation_degrees))

                        start_idx = i * 10
                        end_idx = start_idx + 10
                        selected_pc_file = next((f for f in all_pc_file[start_idx:end_idx] if f.endswith(f"_r{angle}.npy")), None)
                        selected_img_file = next((f for f in all_img_file[start_idx:end_idx] if f.endswith(f"_r_{angle}.png")), None)

                        if selected_pc_file:
                            selected_pc_files.append(selected_pc_file)
                        if selected_img_file:
                            selected_img_files.append(selected_img_file)

                    assert len(selected_pc_files) == len(selected_img_files), f"Mismatch in selected files for category {cate}, subdir {sub}, pc: {len(selected_pc_files)}, img: {len(selected_img_files)}"
                    all_pc_files.append(selected_pc_files)
                    all_img_files.append(selected_img_files)

                except AssertionError as e:
                    print(f"Error in subdirectory {sub} for category {cate}: {e}")
                except FileCheckError as e:
                    print(f"File check failed in subdirectory {sub} for category {cate}: {e}")
                except Exception as e:
                    print(f"Unexpected error in subdirectory {sub} for category {cate}: {e}")

            main_data_list = []
            all_data_list = []
            all_label_list = []
            main_label_list = []

            for i in tqdm(range(len(main_pc_files)), desc=f"Loading train {cate} data"):
                try:
                    main_data = {}
                    main_pc = os.path.join(self.main_pc_path, cate, main_pc_files[i])
                    main_img = os.path.join(self.main_img_path, cate, main_img_files[i])

                    self.check_pc_img(main_pc_files[i], main_img_files[i])

                    main_data["pc"] = main_pc
                    main_data["img"] = main_img

                    all_data = {'pc': [], 'img': []}
                    all_label = []
                    for j, sub in enumerate(sub_dirs):
                        sub_pc_dir = os.path.join(self.all_pc_path, cate, sub)
                        sub_img_dir = os.path.join(self.all_img_path, cate, sub)
                        all_pc = all_pc_files[j][i]
                        all_img = all_img_files[j][i]

                        self.check_pc_img(all_pc, all_img)
                        self.check_main_all(main_pc_files[i], all_pc)

                        all_data['pc'].append(os.path.join(sub_pc_dir, all_pc_files[j][i]))
                        all_data['img'].append(os.path.join(sub_img_dir, all_img_files[j][i]))
                        all_label.append(self.all_category_map[f"{cate}_{sub}"])

                    main_label_list.append(self.main_category_map[cate])
                    main_data_list.append(main_data)
                    all_data_list.append(all_data)
                    all_label_list.append(all_label)

                except FileCheckError as e:
                    print(f"Error checking files for category {cate}: {e}")
                except AssertionError as e:
                    print(f"Assertion failed for category {cate}: {e}")
                except Exception as e:
                    print(f"Unexpected error for category {cate}: {e}")

            return main_data_list, main_label_list, all_data_list, all_label_list

        except Exception as e:
            print(f"Error processing category {cate}: {e}")
            return [], [], [], []

    def load_data_form_base(self):
        main_data_list = []
        main_label_list = []
        all_data_list = []
        all_label_list = []
        if self.task_id == -1:
            cates = ["jar","bottle","laptop","bookshelf","knife"]
        else:
            cates = sorted(os.listdir(self.main_pc_path))
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.process_category, cate): cate for cate in cates}

            for future in as_completed(futures):
                cate = futures[future]
                try:
                    main_data, main_label, all_data, all_label = future.result()
                    main_data_list.extend(main_data)
                    main_label_list.extend(main_label)
                    all_data_list.extend(all_data)
                    all_label_list.extend(all_label)
                except Exception as e:
                    print(f"Error processing category {cate}: {e}")

        return main_data_list, main_label_list, all_data_list, all_label_list

    def load_data_for_increment(self):
        with open(os.path.join(self.index_file, f"train/session_{self.task_id}.txt")) as f:
            paths = f.readlines()
        main_label_list = []
        main_data_list = []
        for file_path in paths:
            category, name = file_path.strip().split("/")[-2:]
            file_path = os.path.join(category, name).replace(".npy", "")
            main_pc_path = os.path.join(self.pc_all_path, file_path + ".npy")
            main_label_list.append(self.increment_category_map[category])
            main_data_list.append(main_pc_path)
        return main_data_list, main_label_list

    def load_prompt(self):
        res = {}
        for main_cate in os.listdir(self.main_text_path):
            main_prompt_path = os.path.join(self.main_text_path, main_cate, "prompt.txt")
            with open(main_prompt_path, "r") as f:
                main_prompt = f.readline()

            tokenized_caption = [self.tokenizer(main_prompt)]
            res[self.main_category_map[main_cate]] = ops.stack(tokenized_caption)

            all_prompt_dir = os.path.join(self.all_text_path, main_cate)
            for sub in os.listdir(all_prompt_dir):
                all_prompt_path = os.path.join(all_prompt_dir, sub, "prompt.txt")
                with open(all_prompt_path, "r") as f:
                    all_prompt = f.readline()

                tokenized_caption = [self.tokenizer(all_prompt)]
                res[self.all_category_map[f"{main_cate}_{sub}"]] = ops.stack(tokenized_caption)
        return res

    def check_seq(self):
    
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        
        output_file_path = os.path.join(self.check_dir, f"check_seq_shuffle_{current_time}.txt")

        with open(output_file_path, 'w') as f:
            for idx in range(len(self.main_data_list)):
                if idx % 5 == 0:
                    
                    f.write("\n")
                    pc_path = self.main_data_list[int(idx / 5)]["pc"]
                    img_path = self.main_data_list[int(idx / 5)]["img"]
                    label = self.main_label_list[int(idx / 5)]

                    f.write(f"main_{int(idx / 5)}-")
                    f.write(f"PC Path: {pc_path}-")
                    f.write(f"Image Path: {img_path}-")
                    f.write(f"Label: {label}")
                    f.write(" | ")
                else:
                    pc_path = self.all_data_list[int(idx / 5)]["pc"][(idx % 5) - 1]
                    img_path = self.all_data_list[int(idx / 5)]["img"][(idx % 5) - 1]
                    label = self.all_label_list[int(idx / 5)][(idx % 5) - 1]

                    f.write(f"all_{int(idx / 5)}_{(idx % 5) - 1}-")
                    f.write(f"PC Path: {pc_path}-")
                    f.write(f"Image Path: {img_path}-")
                    f.write(f"Label: {label}-")
                    f.write(" | ")

        print(f"Paths successfully written to {output_file_path}")

    def shuffle(self):
        print("shuffling data...")
        if self.task_id <= 0:
            combined = list(zip(self.main_data_list, self.main_label_list, self.all_data_list, self.all_label_list))
            random.shuffle(combined)
            self.main_data_list, self.main_label_list, self.all_data_list, self.all_label_list = zip(*combined)

    def __getitem__(self, idx):
        if self.task_id == 0 or self.task_id == -1:
            if idx % 5 == 0:
                pc_data = self.process_pc(self.main_data_list[int(idx / 5)]["pc"])
                img_data = self.process_img(self.main_data_list[int(idx / 5)]["img"])
                return pc_data, img_data, self.prompt_dic[self.main_label_list[int(idx / 5)]], self.main_label_list[int(idx / 5)]
            else:
                pc_data = self.process_pc(self.all_data_list[int(idx / 5)]["pc"][(idx % 5) - 1])
                img_data = self.process_img(self.all_data_list[int(idx / 5)]["img"][(idx % 5) - 1])
                return pc_data, img_data, self.prompt_dic[self.all_label_list[int(idx / 5)][(idx % 5) - 1]], self.all_label_list[int(idx / 5)][(idx % 5) - 1]
        elif self.task_id == -2:
            pc_data = self.process_pc(self.main_data_list[int(idx)]["pc"])
            return pc_data, self.main_label_list[idx]
        else:
            pc_data = self.process_pc(self.main_data_list[idx])
            return pc_data, self.main_label_list[idx]

    def __len__(self):
        return 5 * len(self.main_data_list) if self.task_id == 0 else len(self.main_data_list)

class ShapeNetTest:

    def __init__(self,task_id):
        path = shapenet_test_path
        self.indexfile = path.Index_File
        self.pc_path = path.PC_Path
        self.task_id = int(task_id)
        self.npoints = config.npoints
        self.cache_dir = "data/shapenet/cache/test"
        self.cate_map = main_category_map
        self.check_dir = "log/check"
        self.label_name = shapenet_label
        self.list_of_points = []
        self.list_of_labels = []

        os.makedirs(self.cache_dir, exist_ok=True)

        if self.task_id == -1:
            pc, label = self.load_data_from_source()
            self.list_of_points.extend(pc)
            self.list_of_labels.extend(label)
        else:
            if self.task_id > 0:
                for i in range(self.task_id):
                    cache_file = os.path.join(self.cache_dir, f"task_{i}.pkl")
                    with open(cache_file, 'rb') as f:
                        pc,label = pickle.load(f)
                        self.list_of_points.extend(pc)
                        self.list_of_labels.extend(label)

            cache_file = os.path.join(self.cache_dir, f"task_{self.task_id}.pkl")
            if os.path.exists(cache_file):
                print(f"Loading shapenet-test task {self.task_id} data from cache...")
                with open(cache_file, 'rb') as f:
                    pc, label = pickle.load(f)
                    self.list_of_points.extend(pc)
                    self.list_of_labels.extend(label)
            else:
                print(f"Loading shapenet-test task {self.task_id} data from original source...")
                pc, label = self.load_data_from_source()
                with open(cache_file, 'wb') as f:
                    pickle.dump((pc,label), f)
        

    def load_data_from_source(self):
        task_id = self.task_id if self.task_id != -1 else 0
        with open(self.indexfile + f'/test/session_{task_id}.txt', "r") as f:
            self.datapath = f.readlines()

        list_of_points = []
        list_of_labels = []

        def process_data(index):
            fn = self.datapath[index].strip().split('/')[-1].strip()
            cate = self.datapath[index].split('/')[-2]
            cls = self.cate_map[cate]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.load(os.path.join(self.pc_path, cate, fn)).astype(np.float32)
            point_set = point_set[0:self.npoints, :]
            if self.task_id == -1:
                if cls.item() == 20:
                    list_of_points.append(point_set)
                    list_of_labels.append(cls)
            else:
                list_of_points.append(point_set)
                list_of_labels.append(cls)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(process_data, range(len(self.datapath))), total=len(self.datapath), desc="Loading test data"))


        return list_of_points, list_of_labels

    def check_seq(self):
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        
        output_file_path = os.path.join(self.check_dir, f"shapenet_test_check_seq_{current_time}.txt")

        with open(output_file_path, 'w') as f:
            for idx in range(len(self.list_of_points)):
                points = self.list_of_points[int(idx)]
                label = self.list_of_labels[int(idx)]

                f.write("\n")
                f.write(f"index_{int(idx)}-")
                f.write(f"Points Shape: {points.shape}-")
                f.write(f"Label: {label}")
                f.write(" | ")

        print(f"Check information successfully written to {output_file_path}")

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        
        point_set = point_set.copy()

        if self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3]

        return point_set, label

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        current_points = points[pt_idxs].copy()
        current_points = mindspore.Tensor(current_points).float()
        label_name = self.label_name[int(label)]

        return current_points, label, label_name


# class ModelNet(data.Dataset):
#     def __init__(self, config):
#         self.root = config.DATA_PATH
#         self.npoints = config.npoints
#         self.use_normals = config.USE_NORMALS
#         self.num_category = config.NUM_CATEGORY
#         self.process_data = True
#         self.uniform = True
#         self.generate_from_raw_data = False
#         split = config.subset
#         self.subset = config.subset
#
#         if self.num_category == 10:
#             self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
#         else:
#             self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
#
#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#
#         shape_ids = {}
#         if self.num_category == 10:
#             shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
#             shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
#         else:
#             shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
#             shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
#
#         assert (split == 'train' or split == 'test')
#         shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
#         self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
#                          in range(len(shape_ids[split]))]
#         print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')
#
#         if self.uniform:
#             self.save_path = os.path.join(self.root,
#                                           'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
#         else:
#             self.save_path = os.path.join(self.root,
#                                           'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
#
#         if self.process_data:
#             if not os.path.exists(self.save_path):
#                 # make sure you have raw data in the path before you enable generate_from_raw_data=True.
#                 if self.generate_from_raw_data:
#                     print_log('Processing data %s (only running in the first time)...' % self.save_path,
#                               logger='ModelNet')
#                     self.list_of_points = [None] * len(self.datapath)
#                     self.list_of_labels = [None] * len(self.datapath)
#
#                     for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
#                         fn = self.datapath[index]
#                         cls = self.classes[self.datapath[index][0]]
#                         cls = np.array([cls]).astype(np.int32)
#                         point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
#
#                         if self.uniform:
#                             point_set = farthest_point_sample(point_set, self.npoints)
#                             print_log("uniformly sampled out {} points".format(self.npoints))
#                         else:
#                             point_set = point_set[0:self.npoints, :]
#
#                         self.list_of_points[index] = point_set
#                         self.list_of_labels[index] = cls
#
#                     with open(self.save_path, 'wb') as f:
#                         pickle.dump([self.list_of_points, self.list_of_labels], f)
#                 else:
#                     # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
#                     self.save_path = os.path.join(self.root,
#                                                   'modelnet%d_%s_%dpts_fps.dat' % (
#                                                       self.num_category, split, 8192))
#                     print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
#                     print_log(
#                         'since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, then do fps to {} after, the speed is excepted to be slower due to fps...'.format(
#                             self.npoints), logger='ModelNet')
#                     with open(self.save_path, 'rb') as f:
#                         self.list_of_points, self.list_of_labels = pickle.load(f)
#
#             else:
#                 print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
#                 with open(self.save_path, 'rb') as f:
#                     self.list_of_points, self.list_of_labels = pickle.load(f)
#
#         self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
#         with open(self.shape_names_addr) as file:
#             lines = file.readlines()
#             lines = [line.rstrip() for line in lines]
#         self.shape_names = lines
#
#         # TODO: disable for backbones except for PointNEXT!!!
#         self.use_height = config.use_height
#
#     def __len__(self):
#         return len(self.list_of_labels)
#
#     def _get_item(self, index):
#         if self.process_data:
#             point_set, label = self.list_of_points[index], self.list_of_labels[index]
#         else:
#             fn = self.datapath[index]
#             cls = self.classes[self.datapath[index][0]]
#             label = np.array([cls]).astype(np.int32)
#             point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
#
#             if self.uniform:
#                 point_set = farthest_point_sample(point_set, self.npoints)
#             else:
#                 point_set = point_set[0:self.npoints, :]
#
#         if self.npoints < point_set.shape[0]:
#             point_set = farthest_point_sample(point_set, self.npoints)
#
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#         if not self.use_normals:
#             point_set = point_set[:, 0:3]
#
#         if self.use_height:
#             self.gravity_dim = 1
#             height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
#                                                                                  self.gravity_dim:self.gravity_dim + 1].min()
#             point_set = np.concatenate((point_set, height_array), axis=1)
#
#         return point_set, label[0]
#
#     def __getitem__(self, index):
#         points, label = self._get_item(index)
#         pt_idxs = np.arange(0, points.shape[0])  # 2048
#         if self.subset == 'train':
#             np.random.shuffle(pt_idxs)
#         current_points = points[pt_idxs].copy()
#         current_points = torch.from_numpy(current_points).float()
#         label_name = self.shape_names[int(label)]
#
#         return current_points, label, label_name


