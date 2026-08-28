# HalluPrism

This repository contains the code and experiments for our paper:

**HalluPrism: When Multimodal Uncertainty Should Diagnose, Not Decide**

Accepted to **Findings of EMNLP 2026**.

HalluPrism studies uncertainty in multimodal large language models through three behavioral signals:

- **V**: visual-perturbation sensitivity
- **L**: image-removal confidence retention
- **A**: grounding/relation-probe instability

Instead of treating these signals as independent causal sources, we use them together as a **Perturbation-Response Signature**. Our experiments show that this signature can help diagnose different failure behaviors, while directly compressing it into a single abstention score is not always beneficial.

The repository includes the main experimental pipeline, analysis code, identifiability/interventional-entanglement experiments, perturbation-sensitivity controls, and routing experiments used in the paper.

Human-audit results are reported in the paper. Row-level annotator data and adjudication files are not included in this repository.

## Paper

**HalluPrism: When Multimodal Uncertainty Should Diagnose, Not Decide**  
*Findings of EMNLP 2026*

Paper link: coming soon.

## Citation

Citation details will be added after the final proceedings version is available.