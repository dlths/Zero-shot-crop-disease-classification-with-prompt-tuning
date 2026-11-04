# Zero-Shot Crop Disease Classification with Prompt Tuning & Alignment

**English** | [한국어](README.ko.md)

> Classifying crop diseases in unseen environments **without any training data**, by applying **test-time prompt tuning and distribution alignment** to a CLIP-based vision-language model.

📄 **Soeun Lee**, Chanho Kim, Byeongkeun Kang. *Zero-Shot Crop Disease Classification Method Using Prompt Alignment*. KIBME Fall Conference 2024 (in Korean, first author)

![framework](assets/framework.png)

## Highlights

- Off-the-shelf CLIP reaches only **14.59%** Top-1 accuracy because agricultural images differ greatly from its pre-training data.
- Applying **TPT (test-time prompt tuning) + PromptAlign (distribution alignment)** on MaPLe multi-modal prompts, together with enriched class prompts, raises accuracy to **44.13% Top-1 / 83.00% Top-5**.
- All results are obtained **purely at inference time**, without training on any disease data.

## Motivation

Crop disease classifiers are usually trained on specific datasets, so their accuracy drops sharply on new diseases or under different climate and soil conditions. Collecting and labeling data for every disease and environment is expensive. This calls for **zero-shot learning**, which classifies unseen classes from prior knowledge, and for methods that adapt to distribution shifts at test time.

## Method

1. **Multi-modal prompts (MaPLe)** — Learnable prompt tokens are inserted into both the text and image encoders of CLIP ViT-B/16 so that both modalities are adapted together.
2. **Test-time prompt tuning (TPT)** — For each test image, 64 augmented views are generated, and the prompts are updated to minimize the prediction entropy of the most confident views.
3. **Prompt alignment (PromptAlign)** — The mean and variance of token embeddings at each layer of the image encoder are aligned with pre-computed source statistics using an L1 loss, reducing the distribution shift between datasets.
   The final loss is `L = L_entropy + β · L_align` (β = 100).
4. **Class prompt enrichment**
   - **Expanding "Normal"**: a single word "Normal" for every healthy crop is semantically vague. We add 23 related words (e.g., "Robust", "Energetic", "Alive") and sum their probabilities into the Normal class.
   - **Splitting disease names**: multi-word disease names such as "Cabbage Soft Rot" are also encoded word by word, and their probabilities are merged into the corresponding class.

## Experiments

- **Dataset**: [AI Hub Field Crop Disease Diagnosis Images](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=147) — 10 crops, ~350k images, 21 classes (1 normal + 20 diseases). Only the validation split is used.
- **Setup**: MaPLe on CLIP ViT-B/16 (2 prompt tokens per encoder, 3 layers), AdamW, learning rate 0.04

## Results

| Method | Acc-1 | Acc-5 |
|---|---:|---:|
| CLIP (zero-shot) | 14.59 | 38.25 |
| + Prompt tuning + Prompt alignment | 16.75 | 40.58 |
| + Additional "Normal" prompts | 44.13 | 81.94 |
| + Disease name splitting | **44.13** | **83.00** |

- Plain CLIP performs poorly due to the large gap between agricultural images and its pre-training data.
- The biggest gain comes from **enriching the meaning of the "Normal" class**, showing that prompt design is critical for zero-shot performance.

## Related Work

- Chanho Kim, **Soeun Lee**, Taehyung Lee, Byeongkeun Kang. *Zero-Shot Crop Disease Classification Using Contrastive Language-Image Pre-Trained Model*. KIBME Fall Conference 2023 — the predecessor of this work, which first proposed the "Normal" prompt expansion.
- A tool that converts the same dataset to COCO format: [convert-to-coco](https://github.com/dlths/convert-to-coco)

## Tech Stack

Python · PyTorch · CLIP (ViT-B/16) · MaPLe · TPT · PromptAlign

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
- Khattak et al., *MaPLe: Multi-modal Prompt Learning*, CVPR 2023
- Shu et al., *Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models*, NeurIPS 2022
- Abdul Samadh et al., *Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization*, NeurIPS 2023

## Acknowledgment

This work was supported by the Korea Evaluation Institute of Industrial Technology (KEIT) grant funded by the Korea government (MOTIE) (No. 20018635, Development of smart farm management system and growth monitoring).
