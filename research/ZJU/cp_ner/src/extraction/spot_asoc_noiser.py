from . import constants
from dataclasses import dataclass
import numpy as np


@dataclass
class SpotAsocNoiser:
    spot_noise_ratio: float = 0.1
    asoc_noise_ratio: float = 0.1
    null_span: str = constants.null_span

    def random_insert_spot(self, spot_asoc, spot_label_list=None):
        """随机插入 Spot，类别从 spot_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            spot_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if spot_label_list is None or len(spot_label_list) == 0:
            return spot_asoc
        random_num = sum(np.random.binomial(1, self.spot_noise_ratio, len(spot_asoc)))
        for _ in range(random_num):
            random_position = np.random.randint(low=0, high=len(spot_asoc))
            random_label = np.random.choice(spot_label_list)
            spot_asoc.insert(
                random_position,
                {"span": self.null_span, "label": random_label, 'asoc': list()}
            )
        return spot_asoc

    def random_insert_asoc(self, spot_asoc, asoc_label_list=None):
        """随机插入 Asoc，类别从 asoc_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            asoc_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if asoc_label_list is None or len(asoc_label_list) == 0:
            return spot_asoc
        # asoc_sum = sum([len(x['asoc']) for x in spot_asoc])
        spot_sum = len(spot_asoc)
        random_num = sum(np.random.binomial(1, self.asoc_noise_ratio, spot_sum))
        for _ in range(random_num):
            random_label = np.random.choice(asoc_label_list)
            spot_position = np.random.randint(low=0, high=len(spot_asoc))
            asoc_position = np.random.randint(low=0,
                                              high=len(spot_asoc[spot_position]['asoc']) + 1)
            spot_asoc[spot_position]['asoc'].insert(
                asoc_position,
                (random_label, self.null_span)
            )
        return spot_asoc

    def add_noise(self, spot_asoc, spot_label_list, asoc_label_list):
        spot_asoc = self.random_insert_asoc(
            spot_asoc=spot_asoc,
            asoc_label_list=asoc_label_list,
        )
        spot_asoc = self.random_insert_spot(
            spot_asoc=spot_asoc,
            spot_label_list=spot_label_list,
        )
        return spot_asoc
