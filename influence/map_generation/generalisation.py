import os
from map_generation.options.test_options import TestOptions
from map_generation.util import util
import ntpath

from map_generation.options.base_options import BaseOptions
from map_generation.util.CustomDataLoader import CustomDatasetDataLoader

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 0
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

abs_file = os.path.abspath(__file__)	# 获取model.py文件的绝对路径
# 找到绝对路径的同级目录
abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]

def save_image(visuals, img_path, save_dir):
    generate_imgs = visuals['fake']
    im = util.tensor2im(generate_imgs)
    short_path = ntpath.basename(img_path)
    name = os.path.splitext(short_path)[0]
    img_name = '%s_fake_B.png' % name
    save_path = os.path.join(save_dir, img_name)

    util.save_image(im, save_path, aspect_ratio=opt.aspect_ratio)


baseOption = BaseOptions()


def create_model(opt):
    model = baseOption.find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def process_generalisation(img_folder_path, save_dir):
    opt.dataroot = img_folder_path
    opt.no_dropout = True
    opt.results_dir = save_dir

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()[0]  # get image paths
        # 保存图片
        save_image(visuals, img_path, save_dir)
