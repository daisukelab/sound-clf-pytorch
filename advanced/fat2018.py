"""Multi-fold Freesound Audio Tagging Retry-solution

"""

from src.libs import *
import datetime
from src.metric_fat2018 import eval_fat2018, eval_fat2018_by_probas
from src.models import resnetish18, VGGish
#from src.slack_bot import post_to_slack
def post_to_slack(message):
    print(message)


def get_transforms(cfg):
    NF = cfg.n_mels
    NT = cfg.unit_length
    augs = []
    for a in cfg.aug.split('x'):
        if a == 'RC':
            augs.append(GenericRandomResizedCrop((NF, NT), scale=(0.8, 1.0), ratio=(NF/(NT*1.2), NF/(NT*0.8))))
        elif a == 'SA':
            augs.append(AT.FrequencyMasking(NF//10))
            augs.append(AT.TimeMasking(NT//10))
        else:
            if a:
                raise Exception(f'unknown: {a}')
    return VT.Compose(augs)


def get_model(cfg, num_classes):
    if cfg.model == 'R18':
        return resnetish18(num_classes)
    if cfg.model == 'VGG':
        return VGGish(num_classes)
    raise Exception(f'unknown: {cfg.model}')


def read_metadata(cfg):
    # Make lists of filenames and labels from meta files
    filenames, labels = {}, {}
    for split, npy_folder, meta_filename in [['train', 'work/FSDKaggle2018.audio_train', 'train_post_competition.csv'],
                                             ['test', 'work/FSDKaggle2018.audio_test', 'test_post_competition_scoring_clips.csv']]:
        df = pd.read_csv(cfg.data_root/'FSDKaggle2018.meta'/meta_filename)
        filenames[split] = np.array([(npy_folder + '/' + fname.replace('.wav', '.npy')) for fname in df.fname.values])
        labels[split] = list(df.label.values)

    # Make a list of classes, converting labels into numbers
    classes = sorted(set(labels['train'] + labels['test']))
    for split in labels:
        labels[split] = np.array([classes.index(label) for label in labels[split]])

    return filenames, labels, classes


def calc_stat(cfg, filenames, labels, classes, calc_stat=False, n_calc_stat=10000):
    print(labels)
    class_weight = compute_class_weight('balanced', range(len(classes)), labels['train'])
    class_weight = torch.tensor(class_weight).to(torch.float)

    if calc_stat:
        all_train_lms = np.hstack([np.load(f)[0] for f in filenames['train'][:n_calc_stat]])
        train_mean_std = all_train_lms.mean(), all_train_lms.std()
        print(train_mean_std)
    else:
        train_mean_std = None

    return class_weight, train_mean_std


def run(config_file='config-fat2018.yaml', epochs=None, mixup=None, aug=None, norm=False):
    print(config_file, epochs, mixup, aug)
    cfg = load_config(config_file)
    cfg.epochs = epochs or cfg.epochs
    cfg.mixup = cfg.mixup if mixup is None else mixup
    cfg.aug = aug or cfg.aug or ''
    filenames, labels, classes = read_metadata(cfg)
    class_weight, train_mean_std = calc_stat(cfg, filenames, labels, classes, calc_stat=norm)

    name = datetime.datetime.now().strftime('%y%m%d%H%M')
    name = f'model-{cfg.model}-{cfg.aug}-m{str(cfg.mixup)[2:]}{"-N" if norm else ""}-{name}'
    weight_folder = Path('work/' + name)
    weight_folder.mkdir(parents=True, exist_ok=True)
    results, all_file_probas = [], []
    print(f'Training {weight_folder}')

    test_dataset = LMSClfDataset(cfg, filenames['test'], labels['test'], norm_mean_std=train_mean_std)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.bs, pin_memory=True,
                                              num_workers=multiprocessing.cpu_count())

    skf = StratifiedKFold(n_splits=cfg.n_folds)
    for fold, (train_index, test_index) in enumerate(skf.split(filenames['train'], labels['train'])):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        train_files, val_files = filenames['train'][train_index], filenames['train'][test_index]
        train_ys, val_ys = labels['train'][train_index], labels['train'][test_index]

        train_dataset = LMSClfDataset(cfg, train_files, train_ys, norm_mean_std=train_mean_std,
                                      transforms=get_transforms(cfg))
        valid_dataset = LMSClfDataset(cfg, val_files, val_ys, norm_mean_std=train_mean_std)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, pin_memory=True,
                                                   num_workers=multiprocessing.cpu_count())
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.bs, pin_memory=True,
                                                   num_workers=multiprocessing.cpu_count())

        model = get_model(cfg, len(classes))
        dataloaders = [train_loader, valid_loader, None]
        learner = LMSClfLearner(model, dataloaders, mixup_alpha=cfg.mixup, weight=class_weight)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc')
        trainer = pl.Trainer(gpus=1, max_epochs=cfg.epochs, callbacks=[checkpoint])
        trainer.fit(learner)

        learner.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

        (acc, MAP3), file_probas = eval_fat2018(model, device, test_loader, debug_name='test')
        all_file_probas.append(file_probas)
        results.append(acc)

        fold_weight = weight_folder/Path(checkpoint.best_model_path).name
        copy_file(checkpoint.best_model_path, fold_weight)
        print(f'Saved fold#{fold} weight as {fold_weight}')

    mean_file_probas = np.array(all_file_probas).mean(axis=0)
    acc, MAP3 = eval_fat2018_by_probas(mean_file_probas, labels['test'], debug_name='test')
    np.save(weight_folder/'ens_probas.npy', mean_file_probas)
    report = f'{name},{epochs},{aug},{mixup},{norm},{acc},{np.mean(results)}'
    post_to_slack(report)


if __name__ == '__main__':
    fire.Fire(run)


