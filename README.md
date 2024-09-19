# MimicGen

<p align="center">
  <img width="95.0%" src="docs/images/mimicgen.gif">
</p>

This repository contains the official release of data generation code, simulation environments, and datasets for the [CoRL 2023](https://www.corl2023.org/) paper "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations". 

The released datasets contain over 48,000 task demonstrations across 12 tasks and the MimicGen data generation tool can create as many as you'd like.

Website: https://mimicgen.github.io

Paper: https://arxiv.org/abs/2310.17596

Documentation: https://mimicgen.github.io/docs/introduction/overview.html

For business inquiries, please submit this form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

-------
## Latest Updates
- [09/19/2024] **v1.0.1**: Datasets hosted only on Hugging Face (Google Drive deprecated)
- [07/09/2024] **v1.0.0**: Full code release, including data generation code
- [04/04/2024] **v0.1.1**: Dataset license changed to [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/), which is less restrictive (see [License](#license))
- [09/28/2023] **v0.1.0**: Initial code and paper release

-------

## Useful Documentation Links

Some helpful suggestions on useful documentation pages to view next:

- [Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html)
- [Launching Several Data Generation Runs](https://mimicgen.github.io/docs/tutorials/launching_several.html)
- [Reproducing Published Experiments and Results](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html)
- [Data Generation for Custom Environments](https://mimicgen.github.io/docs/tutorials/datagen_custom.html)
- [Overview of MimicGen Codebase](https://mimicgen.github.io/docs/modules/overview.html)

## Troubleshooting

Please see the [troubleshooting](https://mimicgen.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or submit an issue on our github page.

## License

The code is released under the [NVIDIA Source Code License](https://github.com/NVlabs/mimicgen/blob/main/LICENSE) and the datasets are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Please cite [the MimicGen paper](https://arxiv.org/abs/2310.17596) if you use this code in your work:

```bibtex
@inproceedings{mandlekar2023mimicgen,
    title={MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations},
    author={Mandlekar, Ajay and Nasiriany, Soroush and Wen, Bowen and Akinola, Iretiayo and Narang, Yashraj and Fan, Linxi and Zhu, Yuke and Fox, Dieter},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023}
}
```
