# TLScope

Official code repository for the paper:

**TLScope: A Deep Learning Framework for Quantifying Tertiary Lymphoid Structures from H&E Images Reveals Prognostic Heterogeneity Across Breast Cancer Subtypes**

## Overview

TLScope is a deep learning framework designed for automated detection and quantification of Tertiary Lymphoid Structures (TLS) in H&E-stained histopathology images. This framework reveals the prognostic heterogeneity of TLS across different breast cancer subtypes.


## Repository Structure

- `hover_net/` - Cell segmentation module
- `segmentation/` - Tissue segmentation module
- `weight/` - Pre-trained model weights
- `tls_filter.py` - TLS filtering and analysis
- `wsi_two_stage_split.py` - Whole slide image processing

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{tlscope2025,
  title={TLScope: A Deep Learning Framework for Quantifying Tertiary Lymphoid Structures from H\&E Images Reveals Prognostic Heterogeneity Across Breast Cancer Subtypes},
  year={2025}
}
```

## License

Please refer to the LICENSE file for details.
