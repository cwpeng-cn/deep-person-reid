from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import os

from torchreid.utils import read_json

from ..dataset import VideoDataset


class NewSplitPRID2011Gait(VideoDataset):
    """PRID2011.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """
    dataset_dir = 'prid2011_gait2_cropped'
    dataset_url = None

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a'
        )
        self.cam_b_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b'
        )

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        self.test_dirs = test_dirs

        train = self.process_dir(train_dirs, cam1=True, cam2=True)
        query = self.process_dir(test_dirs, cam1=True, cam2=False)
        gallery = self.process_dir(test_dirs, cam1=False, cam2=True)

        super(NewSplitPRID2011Gait, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirnames, cam1=True, cam2=True):

        tracklets = []
        persons = []
        if cam1 and cam2:
            cams = [self.cam_a_dir, self.cam_b_dir]
            for cam in cams:
                for person in os.listdir(cam):
                    person_path = osp.join(cam, person)
                    seq_num = len(os.listdir(person_path))
                    if seq_num >= 20:
                        persons.append(person)

            dirnames = []
            for person in set(persons):
                if person not in self.test_dirs:
                    dirnames.append(person)
            dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

            for dirname in dirnames:
                for cam in cams:
                    person_dir = osp.join(cam, dirname)
                    if osp.exists(person_dir):
                        img_names = glob.glob(osp.join(person_dir, '*.png'))
                        # print(img_names, len(img_names))
                        if len(img_names) == 0:
                            continue
                        img_names = tuple(img_names)
                        pid = dirname2pid[dirname]
                        tracklets.append((img_names, pid, cam == self.cam_b_dir))
            return tracklets

        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets
