from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CellSegDataset(CustomDataset):


    CLASSES = ("background", "neopla", "inflam", "connec", "necros")
    
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

    def __init__(self , **kwargs):
        super(CellSegDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)