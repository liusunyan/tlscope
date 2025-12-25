from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class TLS1kDataset(CustomDataset):
    """NYU Depth V2 dataset.
    """

    CLASSES = ("background", "TLS")

    
    PALETTE = [[0, 0, 0], [255,255,255]]

    def __init__(self , **kwargs):
        super(TLS1kDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)