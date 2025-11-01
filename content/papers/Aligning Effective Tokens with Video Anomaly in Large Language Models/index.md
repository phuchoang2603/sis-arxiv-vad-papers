---
title: Aligning Effective Tokens with Video Anomaly in Large Language Models
type: other
categories:
- Hybrid
summary: Proposes VA-GPT, a multimodal Large Language Model for video 
  anomaly detection and understanding, utilizing effective token selection and 
  generation modules (SETS and TETG) to improve spatial and temporal 
  localization of anomalies. Introduces instruct-following fine-tuning data and 
  cross-domain benchmarks for robustness evaluation.
benchmarks:
- xd-violence
authors:
- Yingxian Chen
- Jiahui Liu
- Ruidi Fan
- Yanwei Li
- Chirui Chang
- Shizhen Zhao
- Wilton W.T.Fok
- Xiaojuan Qi
- Yik-Chung Wu
date: '2023-10-01'
---

## Aligning Effective Tokens with Video Anomaly in Large Language Models

Yingxian Chen 1∗ Jiahui Liu 1∗ Ruidi Fan 1 Yanwei Li 2 Chirui Chang 1 Shizhen Zhao 1 Wilton W.T.Fok 1 Xiaojuan Qi 1† Yik-Chung Wu 1† 1 2

The University of Hong Kong The Chinese University of Hong Kong

{chenyx, liujh, xjqi, ycwu}@eee.hku.hk

∗ equal contributions †
c †
corresponding authors

## Abstract

Understanding abnormal events in videos is a vital and challenging task that has garnered significant attention in a wide range of applications. Although current video understanding Multi-modal Large Language Models (MLLMs) are capable of analyzing general videos, they often struggle to handle anomalies due to the spatial and temporal sparsity of abnormal events, where the redundant information always leads to suboptimal outcomes. To address these challenges, exploiting the representation and generalization capabilities of Vison Language Models (VLMs) and Large Language Models (LLMs), we propose VA-GPT, a novel MLLM designed for summarizing and localizing abnormal events in various videos. Our approach efficiently aligns effective tokens between visual encoders and LLMs through two key proposed modules: Spatial Effective Token Selection (SETS) and Temporal Effective Token Generation (TETG). These modules enable our model to effectively capture and analyze both spatial and temporal information associated with abnormal events, resulting in more accurate responses and interactions. Furthermore, we construct an instruction-following dataset specifically for fine-tuning video-anomaly-aware MLLMs, and introduce a cross-domain evaluation benchmark based on XD-Violence dataset. Our proposed method outperforms existing state-of-the-art methods on various benchmarks.

## 1. Introduction

Detecting and summarizing abnormal events in videos is critical and challenging, and it has garnered considerable attention across multiple research domains and real-world applications, such as security monitoring, video analysis, and crime detection.

Although many traditional methods [4 , 16 , 37 , 43 , 47 , 73 , 86] have been widely explored for video anomaly detection, they exhibit substantial limitations in their effectiveness [22 , 51 , 60 , 66 , 71 , 84]. These limitations manifest in two aspects: 1) Traditional video anomaly detection

Figure 1. Baseline video understanding MLLM feeds forward every visual token (yellow squares) equally to participate in fine-tuning and inference (top row). Different from it, our method focuses on the effective area (unobstructed area in medium video frames) in each frame and select the Spatial Effective Tokens (orange squares) for the LLM (see Section 3.2) (filtered tokens are shown as gray squares). At the same time, we generate anomaly-aware Temporal Effective Tokens (green squares) (see Section 3.3) based on the assigned anomaly scores (denoted as s) of each frame from a pretrained classifier for better temporal localization of anomalies.

![Image](artifacts/image_000000_fc678031b8a7251fd6bb4962023d6f9c8116acb1c77f521e2692f2ba7f4ee6f4.png)

methods [6 , 13 , 57 , 63 , 64 , 86] essentially approach the task as a closed-set detection and classification problem, inherently limiting their ability to achieve a comprehensive understanding and interpretation of anomalies; 2) These methods [2 , 19 , 23 , 65 , 68 , 78] are restricted by a limited vocabulary, making it difficult for them to handle unseen or novel situations effectively.

Recent advancements [1 , 28 , 34 , 36 , 45 , 58] in Vision Language Models (VLMs) and Large Language Models (LLMs) have demonstrated remarkable capabilities in scene understanding and comprehensive analysis. Multimodal Large Language Models (MLLMs), particularly those designed for video understanding [27 , 29 , 31 , 39 , 42 , 74], have achieved significant progress in general video analysis tasks. However, while these models exhibit strong performance in general video understanding, they fall short in accurately detecting and interpreting anomalies.

For mitigating the above challenges, some works [12 , 55 , 67 , 75 , 79] proposed anomaly-aware video MLLMs to better understand the anomalies in videos. Although these models work well for detecting obvious abnormal events, such as fighting or fire, they typically struggle to effectively align abnormal regions with relevant captions which requires addressing spatial redundancy, and accurately identifying abnormal time intervals by mitigating temporal redundancy. This is because these methods treat all latent tokens with equal priority across spatial and temporal dimensions. This leads to performance degradation caused by redundant tokens unrelated to anomalies. However, in most cases, only small regions within a few frames contain the essential information help to identify an anomaly (as shown in Figure 1). Thus, we explore: How can multimodal architectures evolve selective token generation and processing mechanisms to dynamically prioritize anomaly-salient information while preserving comprehensive scene understanding capabilities?

To address the aforementioned issues, we propose a new model named VA-GPT for analyzing various Videos for Abnormal events by aligning effective and accurate tokens with LLMs across both spatial and temporal dimensions. VA-GPT integrates two key components to identify effective visual tokens for alignment while eliminating redundant tokens that could hinder anomaly analysis and distract model from extracting useful information: 1) we develop the Spatial Effective Token Selection (SETS) module for identifying tokens corresponding to regions with challenges for aligning them with LLMs, while filtering out tokens associated with minor dynamics to remove redundancy. This is because we find that abnormal events often result in different visual changes and variations in local areas (see Figure 1); and 2) we propose the Temporal Effective Token Generation (TETG) module which employs a lightweight pre-trained classifier to assign a confidence score to each frame indicating the possibility of containing abnormal events. Then TETG generates efficient tokens with prior knowledge of the temporal information of abnormal events directly in the language space as additional input to the LLMs, effectively enhancing the model's temporal reasoning and understanding abilities about abnormal events.

Furthermore, beyond conventional benchmarks (indomain benchmark), we establish a new cross-domain evaluation protocol that systematically evaluates model robustness with domain shifts. Based on a novel video dataset, XD-Violence [64], we design comprehensive QAs about abnormal events which include different visual contents from our training data and integrate it as a new cross-domain benchmark. Meanwhile, we design temporal-informationoriented QAs on both in- and cross- domain benckmarks for evaluating temporal localization abilities. Comprehensive experiments demonstrate VA-GPT's superiority, achieving state-of-the-art performance in both in-domain anomaly localization and cross-domain generalization scenarios.

The main contributions are summarized as follows:

- We propose VA-GPT, a video-anomaly-aware MLLM for detecting and summarizing anomalies in various videos, which introduces the MLLM to the specific domain of video anomaly understanding.
- We introduce the SETS and TETG, which enable our MLLM to effectively capture both spatial and temporal information in video sequences, resulting in accurate understanding and localization of abnormal events. Meanwhile, we propose a new instruct-following dataset for video anomaly analysis and a comprehensive cross-domain evaluation benchmark for better evaluating the generalization abilities of MLLMs on video anomalies.
- Our extensive experiments demonstrate that our method outperforms existing state-of-the-art methods in various benchmarks, highlighting its effectiveness and potential for practical applications in video anomaly understanding.

## 2. Related Work

## 2.1. Large Language Models (LLMs)

The domain of Natural Language Processing (NLP) has experienced significant progress, particularly with the emergence of Large Language Models (LLMs). The introduction of the Transformer architecture [3 , 18 , 59] is a critical turning point, followed by other influential language models [3 , 10 , 25 , 80] that exhibited remarkable proficiency. Generative Pre-trained Transformers (GPT) [49] brought about a revolution in NLP by employing auto-regressive prediction, establishing itself as a powerful language modeling approach. More recent groundbreaking contributions, such as ChatGPT [45], GPT-4 [1], LaMDA [56] and LLaMA [58], have expanded the horizon even further. These models, trained on extensive textual data, display extraordinary performance in intricate linguistic tasks.

## 2.2. Vision Language Models (VLMs)

Progress in the fields of computer vision and natural language processing has given rise to the development of vision-language models (VLMs) [14 , 21 , 33 , 34 , 50 , 61 , 69]. These models combine visual and linguistic systems to facilitate cross-modal comprehension and reasoning. Notable examples include CLIP [50] which pairs BERT [10] with ViT [11]; BLIP-2 [28] incorporating Vision Transformer features into Flan-T5 [8]; MiniGPT4 [85], connecting BLIP-2 with Vicuna [28 , 48]; PandaGPT [53] , bridging ImageBind [15] with Vicuna. These models excel in tasks like image classification, captioning, and object detection [24 , 53 , 83]. Recent developments in vision-language models have extended into video processing with models

Figure 2. Detailed illustration of our proposed model. When a video is fed into the model, patch embeddings and class embeddings (c.ebd) are extracted from all frames. 1) Based on the difference in patch embeddings between current frame and its neighbour frame, we can get a filter mask to filter out unimportant visual tokens (dashed square ) from current frame's visual tokens , thereby selecting Spatial Effective Tokens that are compressed with a projector with pooling into aligned content token for each frame, meanwhile take attention with text input from users for resulting aligned context token for each frame; 2) Based on class embeddings (c.ebd) of all frames, we use a pre-trained Anomaly-aware Classifier to localize the time period of abnormal events, thereby generating Temporal Effective Tokens to feed forward into the LLM. All of the resulting aligned tokens are fed into the LLM for reasoning and inference of the whole model.

![Image](artifacts/image_000001_d7378ced629908a60a96ebf4ec21aacd834817c73e8f87cfa6f788a80743b4f8.png)

like Video-Chat [29], Video-ChatGPT [42], Otter [26], Valley [39], mPLUG [27], Video-LLaMA [74], and LLaMAVID [31]. These systems enable interactive video querying, enhance comprehension through audio-visual-text alignment, and support comprehensive video analysis. In this paper, we leverage VLMs and LLMs to develop a novel approach for video anomaly understanding.

## 2.3. Video Anomaly Understanding (VAU)

Annotating surveillance video frames is labor-intensive, prompting researchers to explore alternatives: one-class learning [72], unsupervised anomaly detection without annotations [4 , 5 , 16 , 20 , 37 , 38 , 47], or weakly supervised methods using only video-level annotations [6 , 13 , 30 , 57 , 60 , 62 , 63 , 71 , 76]. In one-class learning, Luo et al. developed a ConvLSTM network for normal segment learning [40]. Several researchers employed Auto-Encoder networks to reconstruct normal frame features [5 , 20 , 81], while others implemented memory mechanisms [4 , 16 , 37 , 47] or meta-learning [38] to enhance generalizability. For weakly supervised learning, Tian [57] sed multiple-instance learning to localize anomalous clips. Zhong et al. uutilized graph convolution networks, though with limited generalization capability. [82]. To address this, Ramachandra et al. developed a Siamese network for normal feature learning. Wan et al. and Zaheer et al. [60 , 71]proposed clustering-based frameworks for anomaly identification. Recent studies have introduced new architectures for spatial-temporal feature ensemble learning [6 , 13 , 30 , 57 , 62 , 63 , 76]. However, these methods merely supply anomaly scores during inference, necessitating empirical threshold establishment on test sets to differentiate abnormal events. Recent research has begun exploring MLLMs to enhance models' capabilities in identifying and describing anomalies [12 , 55 , 67 , 75 , 79].

## 3. Method

## 3.1. Overview

Task. Video anomaly understanding MLLMs aims to determine whether an input video contains abnormal events, meanwhile describing and interacting with the temporal localization and the entire process of the detected abnormal events (if has). We train the model with an instruct-following dataset based on abnormal videos [54], so that the model can better align the tokens between visual encoders and LLMs for presenting and generalizing information about abnormal events.

Pipeline. Considering the video understanding MLLM framework as shown in Figure 2, taking a video (contains T frames) as input, a frozen ViT-based [11] visual encoder (CLIP [52]) extracts visual tokens X t from each video frame V t (t = 1, ..., T). For X t = {x t i } i=1,...,N of the current frame, there are N visual tokens corresponding to equal amounts of image patches. Modality alignment converts the processed visual tokens X t into the semantic space of LLMs. At the same time, text prompts are processed and encoded as text tokens into the same semantic space and serve as a part of input to LLMs. Our key design on models consists of (1) selecting Spatial Effective Tokens X ∗t (SET) from X t for each frame participating in fine-tuning and inference instead of X t (see Section 3.2); and (2) generating Temporal Effective Tokens S ∗t (TET) as anomaly-aware temporal priors, participating in inference to facilitate the temporal localization of abnormal events for LLMs (see Section 3.3). In addition, we produce high-quality instruct-following data on abnormal videos and develop a training strategy for it to maximize the effectiveness of our proposed method (see Section 3.4).

## 3.2. Spatial Effective Token Selection (SETS)

In classical video classification tasks, context and relationships are critical. However, in our MLLM setting, beyond leveraging contextual information, the most crucial problem is aligning the visual and language modalities. Therefore, the key aspect of our design is to extract useful information for effectively aligning visual tokens with the LLM. Since text captions primarily describe anomaly events, which occupy only a small portion of the entire video, aligning all visual patterns with text tokens would be unreasonable and computationally heavy. Thus, we are the first to propose a novel token selection method SETS to achieve efficient and effective alignment.

Inter-frame Difference. For a video, we believe that areas with large changes in adjacent frames are more worthy of attention. As illustrated in Figure 2, for each frame V t of the video, we can regard its previous frame V t − 1 as the reference frame for investigating the difference between current timestamp and previous timestamp. Employing DINOv2 [46] as the feature extractor, denoted as FE, we can extract patch embeddings:

<!-- formula-not-decoded -->

where F t , F t − 1 ∈ R N×C are the extracted embeddings (N indicates the number of image patches and C indicates the channels). Thanks to the distinction and stability of the extracted features, we calculate their patch-wise distances as the inter-frame difference map of the current frame:

<!-- formula-not-decoded -->

where dis(·) indicates Manhattan distance [17] and D t ∈ R N indicates the distances between corresponding patch pairs in neighbour frames.

Select Spatial Effective Tokens. According to the interframe difference map D t , we can set up a vector M t = [m t t
1
, m t t
2
, ..., m t N ] to record the difference of each patch, where top K ratio of elements with the largest distance are assigned with the value of 1, and the rest are assigned as 0. Thus we get a mask for filtering and updating the visual tokens as:

<!-- formula-not-decoded -->

where X ∗t contains the selected Spatial Effective Tokens (SET) which are fed into subsequent processing instead of

X t as shown in Figure 2. SET can efficiently isolate the regions highly related to the abnormal events to participate in both fine-tuning and inference.

## 3.3. Temporal Effective Token Generation (TETG)

Anomaly-aware Classifier. We design a simple but effective MLP FA FA for learning whether each frame is related to an abnormal event. For the class embeddings (denoted as z) extracted from the feature encoder, we can split them based on training video caption into normal and anomaly embeddings, denoted as z n and z a , respectively. Thus we can optimize FA using a binary classification loss:

<!-- formula-not-decoded -->

The anomaly-aware classifier predicts whether each frame is related to anomalies in a video, which can bring additional important prior knowledge to LLMs at a very low cost to facilitate its inference.

Generate Temporal Effective Tokens. Since the information drawn from the anomaly-aware classifier is explicit, we can easily project it to LLMs' text token space through natural languages. Based on the prediction results of the anomaly-aware classifier, we select the first and last frames' timestamps with high confidence of containing abnormal events, denoted as &lt;a-start&gt; and &lt;a-end&gt;, respectively. Then we tokenize them with a template as: "Known common crime types are: 'Shooting','Arson','Arrest', ... There is one of the crime types occurring from &lt;a-start&gt; to &lt;a-end&gt;", resulting Temporal Effective Tokens (TET) in the text token space of LLMs. During inference, with the well-trained lightweight anomaly-aware classifier, TET is used as an additional input to participate in the forward process of the LLM to provide prior knowledge about abnormal events temporally (as shown in Figure 2).

## 3.4. Training Strategy

For modality alignment and instruction tuning, we follow the baseline [31] to ensure visual features are well aligned with the language space. In this work, the training strategy can be divided into two stages: 1) Stage One: Fine-tuning with anomaly video data, and 2) Stage Two: Aligning Spatial Effective Tokens with LLMs.

Fine-tuning with Video Anomaly Data. For enhancing the abnormal scene understanding of LLMs, we construct the UCF-crime [70] Question-Answer pairs for finetuning. We also mix different instruction pairs from various sources [32], including text conversations, single/multi-turn visual Question-Answer pairs, and video Question-Answer pairs. Different formats for text, image, and video inputs are adopted, and the image token is randomly inserted at the beginning or end of user input during training. All modules,

Table 1. Comparing on in-domain (UCF-Crime [54]) and the proposed cross-domain (XD-Violence [64]) benchmarks, our method significantly outperforms other models and achieve the state-of-the-art performance (accuracy is detonated as Acc.) on anomaly video understanding and temporal localization with LLMs (Best results are shown in bold).

| Method             | LLM       | n-domain      | n-domain        | Cross-domain   | Cross-domain     |
|--------------------|-----------|---------------|-----------------|----------------|------------------|
| Method             | LLM       | Total Acc.(%) | Temporal Acc.(% | Total Acc.(%)  | Temporal Acc.(%) |
| Video-Chat [29]    | Vicuna-7B | 22.41         | -               | -              | -                |
| Video-ChatGPT [42] | Vicuna-7B | 24.13         | 28.51           | 24.00          | 29.10            |
| Otter [26]         | LLaMa-7B  | 22.41         | 22.17           | 25.20          | 23.80            |
| Valley [39]        | Vicuna-7B | 20.34         | 14.48           | 21.00          | 20.20            |
| mPLUG [27]         | LLaMa-7B  | 22.76         | -               | -              | -                |
| Video-LLaMA2 [7]   | Vicuna-7B | 21.38         | 26.62           | 24.20          | 23.00            |
| Hawkeye [79]       | LLaVA-7B  | 28.60         | 30.00           | 25.30          | 28.50            |
| LLaMA-VID [31]     | Vicuna-7B | 14.83         | 26.70           | 18.80          | 23.60            |
| VA-GPT (Ours)      | Vicuna-7B | 30.69         | 35.00           | 26.20          | 31.02            |

Table 2. We evaluate our method on the MMEval [12] benchmark and show that our method outperforms the related previous method in different aspects.

| Method          |   Description  |   Causes  |   Effect |
|-----------------|----------------|-----------|----------|
| A-Guardian [12] |          79.65 |     58.92 |    50.64 |
| Ours            |          80.83 |     59.55 |    51.08 |

except the frozen visual encoder, are optimized in this stage. After fine-tuning, the LLM has a preferred perception of anomalies, thus ensuring the effectiveness of the temporal effective tokens (see Section 3.3) during inference. More dataset details are illustrated in Section 4 .

Aligning Spatial Effective Tokens with LLMs. For abnormal video scenes, most areas would not be aligned with languages. Therefore, we implement an additional fine-tuning step. This step involves utilizing Spatial Effective Tokens (see Section 3.2) derived from each video frame within the UCF-Crime dataset. By incorporating these tokens, we aim to provide the model with a more refined understanding of the spatial context of anomalies. It also brings efficient optimization, and the alignment here is only designed for very short-term fine-tuning, which can greatly improve the model's ability to detect and understand anomalies.

## 4. Experiments

Datasets. We fine-tune our model on our proposed instructfollowing format [34] training dataset including 4077 videos and 7730 images based on UCF-Crime [54] dataset. We evaluate the models on two anomaly video understanding benchmarks: UCF-Crime [54] for in-domain evaluation and a proposed benchmark designed based on XD-Violence [64] dataset for cross-domain evaluation, respectively. More details are shown in the supplementary.

Benchmarks and Metrics. To evaluate the ability to review videos and identify anomalies, we utilize a video anomaly understanding evaluation from Video-Bench [44] to assess the temporal comprehensive ability, which contains nature language based Question-Answer pairs from UCF-Crime [54] dataset. Meanwhile, in order to evaluate the model's crossdomain video anomaly understanding ability, we contribute Question-Answer pairs as an extra benchmark based on the XD-Violence dataset. These Question-Answer pairs encompass four options, with each presenting the anomaly category and the respective time intervals during which the anomalies transpire. For each benchmark, we design different sets of questions for two evaluations: one is an overall evaluation of abnormal event detection and understanding, and the other is a special evaluation focusing on temporal localization ability, measured by question answering accuracy (denoted as Total Acc. and Temporal Acc., respectively, higher is better).

Implementation Details. For network structure, we incorporate the pre-trained CLIP [52] and DINOv2 [46] as the visual encoder and Qformer [9] as the text decoder. We follow [31] to freeze the encoder during the modality alignment and to optimize the trainable parameters using anomaly videos and instructions for instruction tuning. During the training process, our model utilizes PyTorch on four NVIDIA A100 GPUs. We employ the AdamW optimizer with a batch size of 64. The learning schedule is set to cosine decay, with a learning rate of 2e-5 and a total of 1 epoch.

## 4.1. Main Results

Results on In-domain Dataset. We first evaluate our method on the in-domain dataset, where the test set belongs to the same style and recording mode as the data used for training in Section 3.4. As shown in Table 1, compared with the baseline [31], with fewer visual embedding tokens and temporal effective tokens, our method brings more than double the performance improvement on total accuracy, also brings a significant increase in temporal localization. Driven by our proposed training strategy and designed effective tokens, more pure and effective visual-semantic information

Figure 3. Qualitative results in Question-Answer diagrams, the red circles in the figures correspond to the bold text in the answers. From short video of only a dozen seconds to medium video of longer than one minute and long video of about half an hour, our model can reason well and understand the content.

![Image](artifacts/image_000002_d9fae28b86f6484911947aa7384d4981d71147a651c667a19139ecb331e860d2.png)

Table 3. Ablation studies on Spatial Effective Token Selection (SETS), Temporal Effective Tokens Generation (TETG), and progressive training strategies. At different model training stages, starting from the baseline (w/o fine-tuning, w/o both), we compare the performance of only using SETS (w.SETS), only using TETG (w.TETG), and using both (w.Both) on the UCF-Crime benchmark. Stage One: Only anomaly video fine-tuning. Stage Two: Anomaly video fine-tuning + Fine-tuning with SETS (Best results are shown in bold).

| Method             | Baseline    | Baseline    | Baseline    | Baseline    | Stage One Fine-tuning Sta   | Stage One Fine-tuning Sta   | Stage One Fine-tuning Sta   | Stage Two Fine-tuning   | Stage Two Fine-tuning   | Stage Two Fine-tuning   |
|--------------------|-------------|-------------|-------------|-------------|-----------------------------|-----------------------------|-----------------------------|-------------------------|-------------------------|-------------------------|
|                    | w/o Both    | w.SETS      | w.TETG      | w.Both      | w.SETS                      | w.TETG                      | w.Both                      | w.SETS                  | w.TETG                  | w.Both                  |
| Total Acc. (%)↑    | 14.83       | 24.83       | 23.79       | 25.12       | 25.86                       | 26.10                       | 27.50                       | 29.31                   | 28.96                   | 30.69                   |
| Temporal Acc. (%)↑ | 26.70       | 27.20       | 27.76       | 28.81       | 29.68                       | 30.02                       | 30.77                       | 31.60                   | 33.58                   | 35.00                   |

of abnormal events is efficiently aligned with LLMs and exhibits powerful anomaly video understanding capabilities. At the same time, we conduct fair comparisons with existing video understanding models [26 , 27 , 29 , 39 , 42 , 74] (see Table 1), and we demonstrate competitive performance. It is worth noting that we use the fewest tokens among all methods to achieve the state-of-the-art results on both total and temporal accuracies.

Results on Cross-domain Dataset. For evaluating the robustness and generalization of the models, we additionally design a cross-domain benchmark. We conduct a fair comparison of our method with the baseline [31] and the existing in-domain methods on the proposed cross-domain benchmark. The results presented in Table 1 showcase a substantial performance improvement over existing methods on the cross-domain dataset, underscoring the exceptional generalization and temporal localization capabilities of our methodology. This clear superiority in performance serves as a compelling validation of the robustness and adaptability of our approach across diverse domains.

Interaction with the Model. We take some interactions with our well-trained model for better evaluation. As shown in Figure 3, we demonstrate the performance of our model in addressing various video anomaly understanding challenges. To evaluate the model's effectiveness, we select videos of different durations: short (0 to 1 minute), medium (1 to 30 minutes), and long (over 30 minutes). This variety allows us to thoroughly assess the model's capabilities in handling diverse anomaly understanding scenarios. In the Road Accident video (left side in Figure 3), our method successfully identifies a car driving at high speed and detects people falling, even in low-resolution footage. For the Explosion video (middle in Figure 3), the model accurately predicts the scene and the anomaly in a medium-length video depicting an explosion. In a normal video exceeding 30 minutes (right side in Figure 3), we demonstrate the model's ability

Table 4. We fine-tune comparison models with our proposed UCF instruct-following data and evaluate the performance of these models before and after fine-tuning on the UCF benchmark.

| Method               | LLM       | Fine-tuned    |   Total Acc.(%)↑ |
|----------------------|-----------|---------------|------------------|
| Video-ChatGPT        | Vicuna-7B | -             |            24.13 |
| Video-ChatGPT        | Vicuna-7B | ✓             |            26.23 |
| LLaMA-VID (Baseline) | Vicuna-7B | -             |            14.83 |
| LLaMA-VID (Baseline) | Vicuna-7B | ✓             |            23.1  |
| Ours                 | Vicuna-7B | -             |            25.12 |
| Ours                 | Vicuna-7B | ✓             |            30.69 |

to focus on both global and local information by asking it to summarize the content.

Comparison on Other Benchmark. We additionally compare our method on another benchmark (MMEval [12]) about anomaly video understanding with LLMs from different aspects. We follow a fair evaluation on our proposed model and obtain the quantity results as shown in Table 2 , which shows the superiority of our method.

## 4.2. Ablation Studies

We conduct extensive ablation studies to validate the effectiveness of the key components in our method: Spatial Effective Token Selection (SETS, Section 3.2) and Temporal Effective Tokens Generation (TETG, Section 3.3) on progressive training strategies in Section 3.4 .

Fine-tuning Stages. The utilization of our high-quality UCF instruct-following data has proven to enhance the model's performance. Fine-tuning with this dataset has effectively contributed to a notable accuracy compared with the baseline. As evidenced in Table 3, with our model designing (both SETS and TETG), the total accuracy for anomaly detection only achieves 25.12% without any fine-tuning (denoted as Baseline). With anomaly video fine-tuning (denoted as Stage One Fine-tuning), the accuracy increases to 27.5%. Furthermore, an efficient tuning with SETS (denoted as Stage Two Fine-tuning) can achieve our final performance of 30.69% total accuracy. Temporal accuracy also shows similar increasing patterns with the scaling tuning stages.

Effectiveness of Fine-tuning Data. For a fair comparison, we fine-tune some high-performance comparison models [31 , 42] and compare them with our proposed UCF instructfollowing data. As shown in Table 4, the performance of these comparison models has increased after fine-tuning, which proves the effectiveness of our proposed data. However, their performance is still not as good as our method, which proves the effectiveness of our proposed model.

Effectiveness of SETS. Our proposed SETS demonstrates efficiency in extracting useful abnormal information, leading to performance enhancement. As illustrated in Table 3 , with the SETS, the accuracy reaches 24.83%, 25.86% and 29.31% without fine-tuning, anomaly video fine-tuning and

Table 5. For the sample rate of tokens, K ratios in SETS, we sample the patch tokens with ordered distance at a fixed sample rate (0.5). The ablation indicates too large sampling rates cause too much context noise, and too small sampling rates lose visual information.

| # Sample Rate K    |   0.1  |   0.3  |   0.5  |   0.7  |   0.9 |
|--------------------|--------|--------|--------|--------|-------|
| Total Acc.(%)↑     |  23.61 |  24.83 |  30.69 |  28.67 | 27.27 |
| Temporal Acc.(%)↑  |  29.03 |  29.93 |  35    |  31.23 | 31.03 |

fine-tuning with SETS, respectively, which far exceed the accuracy of the baseline. Its intuitive mechanism for information filtering can be further analyzed with reference to Figure 4. The initial video is often cluttered with irrelevant and deceptive data. For example, in Figure 4, case one illustrates a scenario where the overall structure is complex, yet only a small segment requires attention. The SETS effectively filters out the dynamic features that do not require attention. Similarly, as in case two, the abnormal area is quite small. Our proposed SETS mechanism effectively filters out redundant and irrelevant information, significantly enhancing the model's ability to accurately pinpoint and recognize abnormal situations.

We also conduct the ablation studies on K ratios about SETS as shown in Tabel 5. Too small or too large K ratios will cause performance degradation in both total and temporal accuracy. If K ratios is too small, redundant information will affect the effectiveness of aligning abnormal event information with corresponding captions. In contrast, if K ratios is too large, some important areas will be filtered out, resulting in information loss and suboptimal performance.

Effectiveness of TETG. Our proposed TETG generates tokens directly in the text token space of LLMs as priors for each video, offering robust priors on the temporal aspects of abnormal events. The provision results in performance improvement without the necessity for fine-tuning. As shown in Table 3, the accuracy rises from 14.83% to 23.79%. Fine-tuning with anomaly video and SETS, the results even achieve 26.10% and 30.69% independently, which manifests the effectiveness of TETG. Besides, the integration of SETS and TETG highlights the importance of leveraging spatial and temporal information effectively in anomaly detection systems, boosting the results to 25.15%, 27.50% and 30.69%, respectively.

## 5. Discussion

Key tokens play key roles. To the best of our knowledge, we are the first to explore how to assign different learnable knowledge to different tokens for better alignment with LLMs on visual contents, thus promoting video anomaly detection and understanding (see Table 1 and Figure 3). We assign the most effective roles to different tokens in both spatial and temporal dimensions, enabling the model to handle

Figure 4. Visualization of the initial videos and our masked results. These two cases illustrate road accident scenarios: one occurring in a bustling street and the other in an empty suburb. Our SETS effectively filters redundant and irrelevant regions (with black patch-level masks).

![Image](artifacts/image_000003_698c2444aa874ca7c6210c9bdeaaca4f84cd27bc63b9d0c3234f1cfd2ae96ce9.png)

various tokens more efficiently. The video contains abundant but redundant information. Our proposed SETS and TETG effectively compress the spatial and temporal information of abnormal events respectively, and utilize the existing alignment mechanism of MLLMs at a very low cost to participate in LLMs' reasoning (see Table 3). Our exploration inspires more representation learning of MLLMs to facilitate downstream tasks.

Data matters a lot. We construct instruct-following data for anomaly videos, containing approximately 4,000 videos, which is significantly less than the amount of baseline finetuning video data (e.g., over 90k videos for fine-tuning in baseline model [31]). We still achieve promising performance on both in-domain and cross-domain benchmarks (see Table 1). This relies on the high-quality Question-Answer pairs in our instruct-following data. Meanwhile, SETS also improves data quality during fine-tuning: visual areas irrelevant to Question-Answer pairs are filtered out (see Figure 4), which allows for significant performance improvements in the second stage of fine-tuning (see Section 3.4) with very few steps (less than 150 iterations).

Broader impacts. Video anomaly understanding has farreaching implications across various sectors, including security, healthcare, industrial safety, and so on. By enhancing the ability to automatically identify and respond to unusual or suspicious activities in real-time, LLMs can significantly improve public safety, crime prevention, patient monitoring, hazard detection, loss prevention, traffic management, and urban planning. These systems offer substantial benefits in terms of operational efficiency and safety.

Limitations. Although our model adeptly portrays the occurrence, type, and area of video abnormal events, it still faces challenges in detecting and describing certain complex scenes. Our strategy represents an early successful validation and investigation of large models for video anomaly identification and localization. Consequently, our method possesses significant potential for enhancement in recognizing diverse abnormal video scenes. These insights motivate us to continue pursuing more powerful and efficient video anomaly understanding technologies in the future, aiming to address more challenges in the real world [35 , 41 , 77].

## 6. Conclusions

In this paper, we propose a novel MLLM for understanding anomalies in videos with LLMs by aligning effective tokens in both temporal and spatial space. The proposed method includes Spatial Effective Token Selection (SETS) for identifying abnormal events in small areas of large scenes and Temporal Effective Tokens Generation (TETG) for addressing the sparseness of abnormal events in video time sequences. We also develop instruct-following data of video anomaly detection to fine-tune the model. Besides, evaluation on the video anomaly understanding benchmark and a proposed cross-domain benchmark demonstrates the effectiveness of the proposed method. It further presents a promising approach for video anomaly understanding using MLLMs, showcasing the potential of effective tokens for enhancing video understanding tasks.

## Acknowledgements

This work has been supported in part by Hong Kong Research Grant Council - Early Career Scheme (Grant No.27209621), General Research Fund Scheme (Grant No.17202422, 17212923, 17215025), Theme-based Research (Grant No.T45-701/22-R) and Shenzhen Science and Technology Innovation Commission (SGDX20220530111405040). Part of the described research work is conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1 , 2
- [2] Borislav Antic and Bj ´ ´ orn Ommer. Video parsing for abnormal- ¨ ¨ ity detection. In 2011 International Conference on Computer Vision, pages 2415–2422, 2011. 1
- [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. CoRR, abs/2005.14165, 2020. 2
- [4] Ruichu Cai, Hao Zhang, Wen Liu, Shenghua Gao, and Zhifeng Hao. Appearance-motion memory consistency network for video anomaly detection. Proceedings of the AAAI Conference on Artificial Intelligence, 35(2):938–946, 2021. 1 , 3
- [5] Yunpeng Chang, Zhigang Tu, Wei Xie, and Junsong Yuan. Clustering driven deep autoencoder for video anomaly detection. In Computer Vision – ECCV 2020, 16th European Conference, pages 329–345, 2022. 3
- [6] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. AAAI2023, 2022. 1 , 3
- [7] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 5
- [8] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai,
9. Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022. 2
- [9] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning, 2023. 5
- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. 2
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2 , 3
- [12] Hang Du, Sicheng Zhang, Binzhu Xie, Guoshun Nan, Jiayang Zhang, Junrui Xu, Hangyu Liu, Sicong Leng, Jiangming Liu, Hehe Fan, Dajiu Huang, Jing Feng, Linli Chen, Can Zhang, Xuhuan Li, Hao Zhang, Jianhang Chen, Qimei Cui, and Xiaofeng Tao. Uncovering what, why and how: A comprehensive benchmark for causation understanding of video anomaly, 2024. 2 , 3 , 5 , 7
- [13] JiaChang Feng, FaTing Hong, and WeiShi Zheng. MIST: multiple instance self-training framework for video anomaly detection. In 2021 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1 , 3
- [14] Yuting Gao, Jinfeng Liu, Zihan Xu, Jun Zhang, Ke Li, Rongrong Ji, and Chunhua Shen. Pyramidclip: Hierarchical feature alignment for vision-language model pretraining, 2022. 2
- [15] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all, 2023. 2
- [16] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memoryaugmented deep autoencoder for unsupervised anomaly detection. In IEEE International Conference on Computer Vision (ICCV), 2019. 1 , 3
- [17] Rodolfo Gonzalez and Sandra Palais. A path-building procedure for iterative circuit computers. Technical report, 1962. 4
- [18] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, Z. Yang, Y. Zhang, and D. Tao. A survey on vision transformer. IEEE Transactions on Pattern Analysis &amp; Machine Intelligence, pages 1–1, 2020. 2
- [19] Mahmudul Hasan, Jonghyun Choi, jan Neumann, Amit K Roy-Chowdhury, and Larry Davis. Learning temporal regularity in video sequences. In Proceedings of IEEE Computer Vision and Pattern Recognition, 2016. 1
- [20] Radu Tudor Ionescu, Fahad Shahbaz Khan, Mariana-Iuliana Georgescu, and Ling Shao. Object-centric auto-encoders and dummy anomalies for abnormal event detection in video. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 3
- [21] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision, 2021. 2
- [22] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weaklysupervised video anomaly detection, 2023. 1
- [23] Louis Kratz and Ko Nishino. Anomaly detection in extremely crowded scenes using spatio-temporal motion pattern models. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 1446–1453, 2009. 1
- [24] Weicheng Kuo, Yin Cui, Xiuye Gu, AJ Piergiovanni, and Anelia Angelova. F-vlm: Open-vocabulary object detection upon frozen vision and language models, 2023. 2
- [25] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019. 2
- [26] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning, 2023. 3 , 5 , 6
- [27] Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, Ji Zhang, Songfang Huang, Fei Huang, Jingren Zhou, and Luo Si. mplug: Effective and efficient vision-language learning by cross-modal skip-connections, 2022. 1 , 3 , 5 , 6
- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. 1 , 2
- [29] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2024. 1 , 3 , 5 , 6
- [30] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. Proceedings of the AAAI Conference on Artificial Intelligence, 36(2):1395–1403, 2022. 3
- [31] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models, 2023. 1 , 3 , 4 , 5 , 6 , 7 , 8
- [32] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection, 2023. 4
- [33] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Munan Ning, and Li Yuan. Moellava: Mixture of experts for large vision-language models, 2024. 2
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 1 , 2 , 5
- [35] Jiahui Liu, Chirui Chang, Jianhui Liu, Xiaoyang Wu, Lan Ma, and Xiaojuan Qi. Mars3d: A plug-and-play motionaware model for semantic segmentation on multi-scan 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9372–9381, 2023. 8
- [36] Jiahui Liu, Xin Wen, Shizhen Zhao, Yingxian Chen, and Xiaojuan Qi. Can ood object detectors learn from foundation
38. models? In European Conference on Computer Vision, pages 213–231. Springer, 2024. 1
- [37] Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In Proceedings of the IEEE International Conference on Computer Vision, 2021. 1 , 3
- [38] Yiwei Lu, Frank Yu, Mahesh Kumar Krishna Reddy, and Yang Wang. Few-shot scene-adaptive anomaly detection. In European Conference on Computer Vision, 2020. 3
- [39] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Da Li, Pengcheng Lu, Tao Wang, Linmei Hu, Minghui Qiu, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability, 2023. 1 , 3 , 5 , 6
- [40] Weixin Luo, Wen Liu, and Shenghua Gao. Remembering history with convolutional lstm for anomaly detection. In 2017 IEEE International Conference on Multimedia and Expo (ICME), pages 439–444, 2017. 3
- [41] Xiaoyang Lyu, Chirui Chang, Peng Dai, Yang-Tian Sun, and Xiaojuan Qi. Total-decom: decomposed 3d scene reconstruction with minimal interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20860–20869, 2024. 8
- [42] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models, 2023. 1 , 3 , 5 , 6 , 7
- [43] Trong-Nguyen Nguyen and Jean Meunier. Anomaly detection in video sequence with appearance-motion correspondence. CoRR, abs/1908.06351, 2019. 1
- [44] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models, 2023. 5
- [45] OpenAI. Introducing chatgpt. 2022. 1 , 2
- [46] Maxime Oquab, Timothee Darcet, Th ´ ´ eo Moutakanni, Huy Vo, ´ ´ Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien ´ ´ Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 4 , 5
- [47] Hyunjong Park, Jongyoun Noh, and Bumsub Ham. Learning memory-guided normality for anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14372–14381, 2020. 1 , 3
- [48] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4, 2023. 2
- [49] Alec Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018. 2
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 2
- [51] Bharathkumar Ramachandra, Michael J. Jones, and Ranga Raju Vatsavai. Learning a distance function with a siamese network to localize anomalies in videos. In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 2020. 1
- [52] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, and Marco Fumero. Clip-forge: Towards zero-shot text-to-shape generation. In CVPR, 2022. 3 , 5
- [53] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all, 2023. 2
- [54] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018. 3 , 5
- [55] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and YingCong Chen. Hawk: Learning to understand open-world video anomalies, 2024. 2 , 3
- [56] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. 2
- [57] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W. Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4975–4986, 2021. 1 , 3
- [58] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste ´ ´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien ` ` Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 1 , 2
- [59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. 2
- [60] Boyang Wan, Yuming Fang, Xue Xia, and Jiajie Mei. Weakly supervised video anomaly detection via center-guided discriminative learning. In Proceedings of the IEEE International Conference on Multimedia and Expo, 2020. 1 , 3
- [61] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks, 2023. 2
- [62] Jie Wu, Wei Zhang, Guanbin Li, Wenhao Wu, Xiao Tan, Yingying Li, Errui Ding, and Liang Lin. Weakly-supervised spatio-temporal anomaly detection in surveillance video. In he Thirtieth International Joint Conference on Artificial Intelligence, 2021. 3
- [63] Peng Wu and Jing Liu. Learning causal temporal relation and feature discrimination for anomaly detection. IEEE Transactions on Image Processing, 30:3513–3527, 2021. 1 , 3
- [64] Peng Wu, jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In European Conference on Computer Vision (ECCV), 2020. 1 , 2 , 5
- [65] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection, 2023. 1
- [66] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection, 2024. 1
- [67] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: Reasoning for video anomaly detection with large language models, 2024. 2 , 3
- [68] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection, 2024. 1
- [69] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training, 2021. 2
- [70] Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance videoand-language understanding: New dataset, baselines, and challenges, 2023. 4
- [71] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In European Conference on Computer Vision (ECCV), pages 358–376. Springer, 2020. 1 , 3
- [72] Muhammad Zaigham Zaheer, Arif Mahmood, Muhammad Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection, 2022. 3
- [73] Muhammad Zaigham Zaheer, Jin-Ha Lee, Marcella Astrid, Arif Mahmood, and Seung-Ik Lee. Cleaning label noise with clusters for minimally supervised anomaly detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, June 2020. 1
- [74] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding, 2023. 1 , 3 , 6
- [75] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Xiaonan Huang, Changxin Gao, Shanjun Zhang, Li Yu, and
78. Nong Sang. Holmes-vau: Towards long-term video anomaly understanding at any granularity, 2024. 2 , 3
- [76] Jiangong Zhang, Laiyun Qing, and Jun Miao. Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection. In 2019 IEEE International Conference on Image Processing (ICIP), pages 4030–4034, 2019. 3
- [77] Jiaqi Zhang, Yan Hu, Xiaojuan Qi, Ting Meng, Lihui Wang, Huazhu Fu, Mingming Yang, and Jiang Liu. Polar eyeball shape net for 3d posterior ocular shape representation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 180–190. Springer, 2023. 8
- [78] Bin Zhao, Fei-Fei Li, and Eric P. Xing. Online detection of unusual events in videos via dynamic sparse coding. In CVPR 2011, pages 3313–3320, 2011. 1
- [79] Jianing Zhao, Jingjing Wang, Yujie Jin, Jiamin Luo, and Guodong Zhou. Hawkeye: Discovering and grounding implicit anomalous sentiment in recon-videos via sceneenhanced video large language model. In Proceedings of ACM MM 2024, 2024. 2 , 3 , 5
- [80] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models, 2023. 2
- [81] Yiru Zhao, Bing Deng, Chen Shen, Yao Liu, Hongtao Lu, and Xian-Sheng Hua. Spatio-temporal autoencoder for video anomaly detection. In Proceedings of the 25th ACM International Conference on Multimedia, page 1933–1941, New York, NY, USA, 2017. Association for Computing Machinery. 3
- [82] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H. Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 3
- [83] Lanfeng Zhong, Xin Liao, Shaoting Zhang, Xiaofan Zhang, and Guotai Wang. Vlm-cpl: Consensus pseudo labels from vision-language models for human annotation-free pathological image classification, 2024. 2
- [84] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection, 2023. 1
- [85] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023. 2
- [86] Yi Zhu and Shawn D. Newsam. Motion-aware feature for improved video anomaly detection. In British Machine Vision Conference (BMVC), 2019. 1