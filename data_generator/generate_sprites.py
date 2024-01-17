import argparse

from PIL import Image
import sys, os
import numpy as np
import os
import imageio.v2 as imageio
from utils import read_config

cfg = read_config()
ARGS = cfg['sprites']
ARGS = argparse.Namespace(**ARGS)

# STEP1: generate random characters

def gen_char(body, bottom, top, hair):
    material_dir = ARGS.material_dir

    np.random.seed(seed)

    # then randomly sample the components
    attributes = {'body': str(body), 'bottomwear': str(bottom),
                  'topwear': str(top), 'hair': str(hair)}
    img_list = []
    for attr in ['body', 'bottomwear', 'topwear', 'hair']:
        path = material_dir + attr + '/'
        filename = attributes[attr] + '.png'
        # print path+filename
        img_list.append(Image.open(path + filename))
    # shoes
    img_list.append(Image.open(material_dir + 'shoes/1.png'))

    # then merge all!
    f = Image.new('RGBA', img_list[0].size, 'black')
    for i in range(len(img_list)):
        f = Image.alpha_composite(f, img_list[i].convert('RGBA'))

    # save image
    classname = str(body) + str(bottom) + str(top) + str(hair)  # +str(weapon)
    f.save('%s.png' % classname)

    img = Image.open('%s.png' % classname)
    # crop to 64 * 64
    width = 64
    height = 64
    imgwidth, imgheight = img.size
    N_width = int(imgwidth / width)
    N_height = int(imgheight / height)
    path = ARGS.frame_save_dir
    if not os.path.exists(path):
        os.makedirs(path)

    N_total = 273  # 273 png files in total
    actions = {
        'spellcard': {'back': range(0, 7), 'left': range(13, 20),
                      'front': range(26, 33), 'right': range(39, 46)},
        'thrust': {'back': range(52, 60), 'left': range(65, 73),
                   'front': range(78, 86), 'right': range(91, 99)},
        'walk': {'back': range(104, 113), 'left': range(117, 126),
                 'front': range(130, 139), 'right': range(143, 152)},
        'slash': {'back': range(156, 162), 'left': range(169, 175),
                  'front': range(182, 188), 'right': range(195, 201)},
        'shoot': {'back': range(208, 221), 'left': range(221, 234),
                  'front': range(234, 247), 'right': range(247, 260)},
        'hurt': {'front': range(260, 266)}
    }
    duration = 0.1
    copy_path = ARGS.frame_save_dir

    # create save list
    # selected_actions = ARGS.selected_actions
    selected_actions = ['spellcard', 'walk', 'slash']
    for act in selected_actions:
        if not os.path.exists(path + act + '/'):
            os.makedirs(path + act + '/')

    for j in range(N_height):
        for i in range(N_width):
            ind = j * N_width + i

            # for spellcard
            if ind >= 13 and ind < 46:
                box = (i * width, j * height, (i + 1) * width, (j + 1) * height)
                if ind >= 13 and ind < 20:
                    pose = 'left'
                    k = ind - 13
                    if k == 6:
                        k = k + 1
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'spellcard/' + pose + '_%s_%d.png' % (classname, k))
                    if k == 4:
                        a.save(path + 'spellcard/' + pose + '_%s_%d.png' % (classname, 6))
                if ind >= 26 and ind < 33:
                    pose = 'front'
                    k = ind - 26
                    if k == 6:
                        k = k + 1
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'spellcard/' + pose + '_%s_%d.png' % (classname, k))
                    if k == 4:
                        a.save(path + 'spellcard/' + pose + '_%s_%d.png' % (classname, 6))
                if ind >= 39 and ind < 46:
                    pose = 'right'
                    k = ind - 39
                    if k == 6:
                        k = k + 1
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'spellcard/' + pose + '_%s_%d.png' % (classname, k))
                    if k == 4:
                        a.save(path + 'spellcard/' + pose + '_%s_%d.png' % (classname, 6))

            # for walk
            if ind >= 117 and ind < 152:
                box = (i * width, j * height, (i + 1) * width, (j + 1) * height)
                if ind >= 118 and ind < 126:
                    pose = 'left'
                    k = ind - 118
                    if k == 7:
                        k = 0
                    else:
                        k = k + 1
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'walk/' + pose + '_%s_%d.png' % (classname, k))
                if ind >= 131 and ind < 139:
                    pose = 'front'
                    k = ind - 131
                    if k == 7:
                        k = 0
                    else:
                        k = k + 1
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'walk/' + pose + '_%s_%d.png' % (classname, k))
                if ind >= 144 and ind < 152:
                    pose = 'right'
                    k = ind - 144
                    if k == 7:
                        k = 0
                    else:
                        k = k + 1
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'walk/' + pose + '_%s_%d.png' % (classname, k))

            # for slash
            if ind >= 169 and ind < 201:
                box = (i * width, j * height, (i + 1) * width, (j + 1) * height)
                if ind >= 169 and ind < 175:
                    pose = 'left'
                    k = ind - 169
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, k))
                    if k == 4:
                        a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, 6))
                    if k == 0:
                        a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, 7))
                if ind >= 182 and ind < 188:
                    pose = 'front'
                    k = ind - 182
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, k))
                    if k == 4:
                        a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, 6))
                    if k == 0:
                        a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, 7))
                if ind >= 195 and ind < 201:
                    pose = 'right'
                    k = ind - 195
                    a = img.crop(box)
                    a.convert('RGB')
                    a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, k))
                    if k == 4:
                        a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, 6))
                    if k == 0:
                        a.save(path + 'slash/' + pose + '_%s_%d.png' % (classname, 7))

    # now remove the png files
    os.remove('%s.png' % classname)

# STEP2: read the .png files and generate numpy data files

num_direction = {'front': 0, 'left': 1, 'right': 2, 'back': 3}
n_class = 6
n_frames = 8
n_directions = 3


def load_seq(path, labels, action, direction):
    num = ''
    for i in range(len(labels)):
        num = num + str(labels[i])

    # return sequence and label
    seq = []
    for frame in range(n_frames):
        fr = str(frame)
        filename = action + '/' + direction + '_' + num + '_' + fr + '.png'
        im = imageio.imread(path + filename)
        seq.append(np.asarray(im, dtype='f'))

    attr_labels = np.zeros([len(labels), n_class])
    for i in range(len(labels)):
        attr_labels[i, labels[i]] = 1
    di_labels = np.zeros(n_directions)
    di_labels[num_direction[direction]] = 1
    list_attr = []
    list_di = []
    for i in range(n_frames):
        list_attr.append(attr_labels)
        list_di.append(di_labels)

    return np.asarray(seq), np.asarray(list_attr, dtype='f'), \
           np.asarray(list_di, dtype='f')


def save_npy():
    load_path = ARGS.frame_save_dir
    save_path = ARGS.npy_save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    actions = ARGS.selected_actions
    directions = ['front', 'left', 'right']

    seed_list = range(0, n_class ** 4)
    np.random.seed(423)
    seed_list = np.random.permutation(seed_list)

    for action in actions:
        for direction in directions:
            im_npy_file_name = action + '_' + direction + '_frames'
            attr_npy_file_name = action + '_' + direction + '_attributes'
            di_npy_file_name = action + '_' + direction + '_directions'
            im_seq = []
            attr_seq = []
            di_seq = []
            for seed in seed_list:
                body = int(seed / n_class ** 3)
                seed = int(np.mod(seed, n_class ** 3))
                bottom = int(seed / n_class ** 2)
                seed = int(np.mod(seed, n_class ** 2))
                top = int(seed / n_class)
                hair = int(np.mod(seed, n_class))

                labels = [body, bottom, top, hair]

                seq, attr, di = load_seq(load_path, labels, action, direction)
                im_seq.append(seq)
                attr_seq.append(attr)
                di_seq.append(di)
            im_seq = np.asarray(im_seq, dtype='f')[:, :, :, :, :3] / 256.0
            attr_seq = np.asarray(attr_seq, dtype='f')
            di_seq = np.asarray(di_seq, dtype='f')

            # training and test data
            N_train = 1000
            im_seq_train = im_seq[:N_train]
            im_seq_test = im_seq[N_train:]
            np.save(save_path + im_npy_file_name + '_train.npy', im_seq_train)
            np.save(save_path + im_npy_file_name + '_test.npy', im_seq_test)
            print('saved ' + save_path + im_npy_file_name + '_train.npy')
            print('saved ' + save_path + im_npy_file_name + '_test.npy')

            attr_seq_train = attr_seq[:N_train]
            attr_seq_test = attr_seq[N_train:]
            np.save(save_path + attr_npy_file_name + '_train.npy', attr_seq_train)
            np.save(save_path + attr_npy_file_name + '_test.npy', attr_seq_test)
            print('saved ' + save_path + attr_npy_file_name + '_train.npy')
            print('saved ' + save_path + attr_npy_file_name + '_test.npy')

            di_seq_train = di_seq[:N_train]
            di_seq_test = di_seq[N_train:]
            np.save(save_path + di_npy_file_name + '_train.npy', di_seq_train)
            np.save(save_path + di_npy_file_name + '_train.npy', di_seq_test)
            print('saved ' + save_path + di_npy_file_name + '_train.npy')
            print('saved ' + save_path + di_npy_file_name + '_test.npy')


if __name__ == '__main__':
    n_class = 6
    seed_list = range(0, n_class ** 4)
    for seed in seed_list:
        body = int(seed / n_class ** 3)
        seed = int(np.mod(seed, n_class ** 3))
        bottom = int(seed / n_class ** 2)
        seed = int(np.mod(seed, n_class ** 2))
        top = int(seed / n_class)
        hair = int(np.mod(seed, n_class))

        gen_char(body, bottom, top, hair)
        if (seed + 1) % 100 == 0:
            print('generate %d/%d sprite sequences' % (seed + 1, n_class ** 4))

    save_npy()
