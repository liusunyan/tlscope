from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class Tumor_necrosis_Dataset(CustomDataset):
    """NYU Depth V2 dataset.
    """

    CLASSES = ("background", "Tumor", "Necrosis")

    
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

    def __init__(self , **kwargs):
        super(Tumor_necrosis_Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)