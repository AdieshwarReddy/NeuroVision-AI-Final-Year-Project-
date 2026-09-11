# Primary Dataset Specification: Jun Cheng Figshare Brain Tumor Dataset

## Dataset Metadata
- **Official Title**: brain tumor dataset
- **Authors**: Jun Cheng, Wei Huang, Shenghai Tao, Zhiwei Hou, Tao Zhang, Zhen Han, Li Wang, Jinyuan Wu, Hong Gao, Wei Deng, et al.
- **Repository**: Figshare
- **DOI**: [10.6084/m9.figshare.1512427.v5](https://doi.org/10.6084/m9.figshare.1512427.v5)
- **License**: CC BY 4.0
- **Modality**: T1-weighted Contrast-Enhanced Magnetic Resonance Imaging (T1-CE MRI)
- **Acquisition Centers**: Nanfang Hospital, Guangzhou, China and General Hospital, Tianjing Medical University, China (2005–2010).

## Cohort Characteristics & Tumor Categories
- **Total Slices**: 3,064 2D slices (512 $\times$ 512 resolution, 16-bit intensity)
- **Total Patients**: 233 unique patients
- **Class Breakdown**:
  1. **Meningioma** (Label = 1): 708 slices from 82 patients
  2. **Glioma** (Label = 2): 1,426 slices from 89 patients
  3. **Pituitary Tumor** (Label = 3): 930 slices from 62 patients

## MATLAB `.mat` Structure (`cjdata`)
Each individual slice is stored in a MATLAB structure containing:
- `cjdata.label`: 1 (Meningioma), 2 (Glioma), 3 (Pituitary)
- `cjdata.PID`: Patient ID string
- `cjdata.image`: 2D image matrix ($512 \times 512$ uint16/double)
- `cjdata.tumorBorder`: Vector of polygon border coordinates $[x_1, y_1, x_2, y_2, \dots]$
- `cjdata.tumorMask`: 2D binary matrix ($512 \times 512$) indicating ground-truth tumor location annotated by clinical experts.

## Research Task Scope & Safety Boundaries
- **Task**: 3-Class Intracranial Neoplasm Type Differentiation (Meningioma vs. Glioma vs. Pituitary Tumor).
- **Out of Scope**: Population-level screening, non-tumor detection, early detection, or autonomous diagnostic replacement.
