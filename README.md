# Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping

This is a demo code of the proposed method in the following reference:

S. Takemoto, K. Naganuma, and S. Ono,
``Spatio-Spectral-Structure-Tensor-Total-Variation-for-Hyperspectral-Image-Denoising-and-Destriping,''
_IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_, 2025.

Update history:
Jul. 5, 2025: v1.0 

For more information, see the following

- Project website: https://www.mdi.c.titech.ac.jp/publications/s3ttv
- Preprint paper: https://arxiv.org/abs/2404.03313

## How to use
1. **Setting parameters**
 - Choose the image (JasperRidge, PaviaUniversity, or Beltsville)
 - Adjust the parameters
   - `params.rho`: parameter for the radii of the noise terms
   - `params.blocksize`: Block size of spatio-spectral structure tensor
   - `params.stopcri`: Stopping criterion
   - `params.maxiter`: Maximum number of iterations
   - `params.disprate`: Period to display intermediate results
 - Set as `use_GPU` = 1 if you use GPU.
 - Set as `use_fast` = 1 if you use fast convergence version.

2. Run ```main_S3TTV.m```


## Our Reference
If you use this code, please cite the following paper:

```
@misc{Takemoto2024Spatio,
      title={Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping}, 
      author={Takemoto, Shingo and Ono, Shunsuke},
      year={2024},
      eprint={2308.00500},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```


## Comparison with existing methods
This repository also supports comparison with several existing denoising and destriping methods.

### QRNN3D

To evaluate **QRNN3D** [1], follow the steps below:

1. Download the official code from [https://github.com/Vandermode/QRNN3D](https://github.com/Vandermode/QRNN3D?tab=readme-ov-file)

2. Download the fine-tuned checkpoint (Pavia Centre) from [Google Drive](https://drive.google.com/file/d/1o1R3PVZhsJzbpJjRHlhMGp8fH6Ua4XV-/view?usp=drive_link)

3. Run `hsi_test.py`.


### Other conventional methods

To compare with other methods (**SSTV** [2], **HSSTV** [3], **l0-l1HTV** [4], **STV** [5], **SSST** [6], **LRTDTV** [7], **FGSLR** [8], **TPTV** [9], and **FastHyMix** [10]):

1. Download and extract the following repositories, and place each extracted folder into the `compared_methods/` directory:

    - **LRTDTV**  
      [https://github.com/zhaoxile/Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition](https://github.com/zhaoxile/Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition)

    - **FGSLR**  
      [https://chenyong1993.github.io/yongchen.github.io/](https://chenyong1993.github.io/yongchen.github.io/)

    - **TPTV**  
      [https://github.com/chuchulyf/ETPTV](https://github.com/chuchulyf/ETPTV)

    - **FastHyMix**  
      [https://github.com/LinaZhuang/HSI-MixedNoiseRemoval-FastHyMix](https://github.com/LinaZhuang/HSI-MixedNoiseRemoval-FastHyMix)

2. Run `main_with_comparisons.m`
 - Choose the target image (`JasperRidge`, `PaviaUniversity`, or `Beltsville`)
 - Enable each method by setting `"enable"` to `true` in the corresponding section
 - Adjust parameters for each method

3. Run the result visualization script: `plot_result.m`
- Use `show_band` to select the band index for visualization
- The script compares results in `result/<condition>/<name_method>/` and selects the best result for each method



---

## References

```bibtex
[1] K. Wei, Y. Fu, and H. Huang, ``3-D quasi-recurrent neural network for hyperspectral image denoising,'' IEEE Trans. Neural Netw. Learn. Syst., vol. 32, no. 1, pp. 363--375, 2021.

[2] H. K. Aggarwal and A. Majumdar, ``Hyperspectral image denoising using spatio-spectral total variation,'' IEEE Geosci. Remote Sens. Lett., vol. 13, no. 3, pp. 442--446, 2016.

[3] S. Takeyama, S. Ono, and I. Kumazawa, ``A constrained convex optimization approach to hyperspectral image restoration with hybrid spatio-spectral regularization,'' Remote Sens., vol. 12, no. 21, 2020.

[4] M. Wang, Q. Wang, J. Chanussot, and D. Hong, ``$l_0$-$l_1$ hybrid total variation regularization and its applications on hyperspectral image mixed noise removal and compressed sensing,'' IEEE Trans. Geosci. Remote Sens., vol. 59, no. 9, pp. 7695--7710, 2021.

[5] S. Lefkimmiatis, A. Roussos, P. Maragos, and M. Unser, ``Structure tensor total variation,'' SIAM J. Imag. Sci., vol. 8, no. 2, pp. 1090--1122, 2015.

[6] R. Kurihara, S. Ono, K. Shirai, and M. Okuda, ``Hyperspectral image restoration based on spatio-spectral structure tensor regularization,'' in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2017, pp. 488--492.

[7] Y. Wang, J. Peng, Q. Zhao, Y. Leung, X. Zhao, and D. Meng, ``Hyperspectral image restoration via total variation regularized low-rank tensor decomposition,'' IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 11, no. 4, pp. 1227--1243, 2018.

[8] Y. Chen, T. Huang, W. He, X. Zhao, H. Zhang, and J. Zeng, ``Hyperspectral image denoising using factor group sparsity-regularized nonconvex low-rank approximation,'' IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1--16, 2022.

[9] Y. Chen, W. Cao, L. Pang, J. Peng, and X. Cao, ``Hyperspectral image denoising via texture-preserved total variation regularizer,'' IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1--14, 2023.

[10] L. Zhuang and M. K. Ng, ``FastHyMix: Fast and parameter-free hyperspectral image mixed noise removal,'' IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 8, pp. 4702--4716, 2023.