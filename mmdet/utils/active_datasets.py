import mmcv
import numpy as np


def get_init_labeled_set(cfg):
    # load dataset anns
    anns = load_ann_list(cfg.data.train.dataset.ann_file)

    # get all indexes
    all_indexes = np.arange(len(anns[0]) + len(anns[1]))

    # random select labeled set
    np.random.shuffle(all_indexes)
    labeled_set = all_indexes[:cfg.active_learning.init_num].copy()
    unlabeled_set = all_indexes[cfg.active_learning.init_num:cfg.active_learning.init_num*2].copy()
    labeled_set.sort()
    unlabeled_set.sort()

    return labeled_set, unlabeled_set, all_indexes, anns


def create_active_labeled_set(cfg, labeled_set, anns, cycle):
    # split labeled set into 2007 and 2012
    selected_sets_labeled = [
        labeled_set[labeled_set < len(anns[0])],
        labeled_set[labeled_set >= len(anns[0])] - len(anns[0])
    ]

    # create labeled ann files
    active_labeled_path = []
    for ann, selected_set_labeled, year in zip(anns, selected_sets_labeled, ['07', '12']):
        save_folder = cfg.work_dir + '/cycle' + str(cycle)
        mmcv.mkdir_or_exist(save_folder)
        save_path = save_folder + '/trainval_l' + year + '.txt'
        np.savetxt(save_path, ann[selected_set_labeled], fmt='%s')
        active_labeled_path.append(save_path)

    # update cfg
    cfg.data.train.dataset.ann_file = active_labeled_path
    cfg.data.train.times = cfg.l_repeat

    return cfg


def create_active_unlabeled_set(cfg, unlabeled_set, anns, cycle):
    # split labeled set into 2007 and 2012
    selected_sets_unlabeled = [
        unlabeled_set[unlabeled_set < len(anns[0])],
        unlabeled_set[unlabeled_set >= len(anns[0])] - len(anns[0])
    ]

    # create labeled ann files
    active_unlabeled_path = []
    for ann, selected_set_unlabeled, year in zip(anns, selected_sets_unlabeled, ['07', '12']):
        save_folder = cfg.work_dir + '/cycle' + str(cycle)
        mmcv.mkdir_or_exist(save_folder)
        save_path = save_folder + '/trainval_u' + year + '.txt'
        np.savetxt(save_path, ann[selected_set_unlabeled], fmt='%s')
        active_unlabeled_path.append(save_path)

    # update cfg
    cfg.data.train.dataset.ann_file = active_unlabeled_path
    cfg.data.train.times = cfg.u_repeat

    return cfg


def load_ann_list(paths):
    anns = []
    for path in paths:
        anns.append(np.loadtxt(path, dtype='str'))
    return anns


def update_labeled_set(active_metric, all_set, labeled_set, budget):

    values = active_metric.cpu().numpy()
    all_unlabeled_set = np.array(list(set(all_set) - set(labeled_set)))
    unlabeled_values = values[all_unlabeled_set]
    arg = unlabeled_values.argsort()
    new_labeled_set = all_unlabeled_set[arg[-budget:]]

    labeled_set = np.concatenate((labeled_set, new_labeled_set))
    all_unlabeled_set = np.array(list(set(all_set) - set(labeled_set)))
    np.random.shuffle(all_unlabeled_set)
    unlabeled_set = all_unlabeled_set[:labeled_set.shape[0]]
    labeled_set.sort()
    unlabeled_set.sort()

    return labeled_set, unlabeled_set
