# Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping

This is a demo code of the proposed method in the following reference:

S. Takemoto and S. Ono,
``Spatio-Spectral-Structure-Tensor-Total-Variation-for-Hyperspectral-Image-Denoising-and-Destriping.''

Update history:
Apr. 4, 2024: v1.0 

For more information, see the following

- Project website: https://www.mdi.c.titech.ac.jp/publications/s3ttv
- Preprint paper: https://arxiv.org/abs/2404.03313

## How to use
1. **Setting parameters**
 - Choose the image (JasperRidge or PaviaUniversity)
 - Adjust the parameters
   - `params.rho`: parameter for the radii of the noise terms
   - `params.blocksize`: Block size of spatio-spectral structure tensor
   - `params.stopcri`: Stopping criterion
   - `params.maxiter`: Maximum number of iterations
   - `params.disprate`: Period to display intermediate results
 - Set as `use_GPU` = 1 if you use GPU.
 - Set as `use_fast` = 1 if you use fast convergence version.

2. Run ```main_S3TTV.m```


## Reference
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
