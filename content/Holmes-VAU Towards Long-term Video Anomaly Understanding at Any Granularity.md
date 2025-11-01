## Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

Huaxin Zhang 1 Xiaohao Xu 2 Xiang Wang 1 Jialong Zuo 1 Xiaonan Huang 2 Changxin Gao 1 3 Li Yu 1 1*

Shanjun Zhang Nong Sang

1 Key Laboratory of Image Processing and Intelligent Control,

School of Artificial Intelligence and Automation, Huazhong University of Science and Technology 2 University of Michigan, Ann Arbor 3 Kanagawa University

{zhanghuaxin,wxiang,cgao,hustlyu,nsang}@hust.edu.cn, {xiaohaox,xiaonanh}@umich.edu, {chiyoz01}@kanagawa-u.ac.jp

Figure 1. Motivation . Left: Existing datasets lack the hierarchical structure to capture transient and sustained anomalies across varying temporal scales. Our HIVAU-70k dataset addresses this by providing multi-granularity annotations—clip, event, and video levels—that enable detailed anomaly analysis in complex real-world scenarios. Right: Inspired by Sherlock Holmes's knack for zeroing in on critical details, our Holmes-VAU method integrates an Anomaly-focused Temporal Sampler with a multi-modal LLM, directing model attention to anomaly-rich segments, which enables models to decode complex, long-term video anomalies efficiently.

![Image](artifacts/image_000000_0a9138ce0aaaa096817626f52ebaa67be5456fe8f2ed9466537f5874332b44c2.png)

## Abstract

How can we enable models to comprehend video anomalies occurring over varying temporal scales and contexts? Traditional Video Anomaly Understanding (VAU) methods focus on frame-level anomaly prediction, often missing the interpretability of complex and diverse real-world anomalies. Recent multimodal approaches leverage visual and textual data but lack hierarchical annotations that capture both short-term and long-term anomalies. To address this challenge, we introduce HIVAU-70k, a large-scale benchmark for hierarchical video anomaly understanding across any granularity. We develop a semi-automated annotation engine that efficiently scales high-quality annotations by combining manual video segmentation with recursive free-text

* Corresponding author

annotation using large language models (LLMs). This results in over 70,000 multi-granular annotations organized at clip-level, event-level, and video-level segments. For efficient anomaly detection in long videos, we propose the Anomaly-focused Temporal Sampler (ATS). ATS integrates an anomaly scorer with a density-aware sampler to adaptively select frames based on anomaly scores, ensuring that the multimodal LLM concentrates on anomaly-rich regions, which significantly enhances both efficiency and accuracy. Extensive experiments demonstrate that our hierarchical instruction data markedly improves anomaly comprehension. The integrated ATS and visual-language model outperform traditional methods in processing long videos. Our benchmark and model are publicly available at https: //github.com/pipixin321/HolmesVAU .

## 1. Introduction

Video Anomaly Understanding (VAU) is crucial for applications such as video surveillance [46], violent content analysis [56], and autonomous driving [63]. Detecting deviations from normal patterns aids in hazard prevention and real-time decision-making. Traditional methods [14 , 49 , 74] mainly focus on frame-level predefined closed-set anomaly prediction, assigning an anomaly score to each frame. However, these approaches often fail to describe and understand complex anomalies in the real world.

To address this gap, open-world anomaly understanding [57] embraces the diversity and unpredictability of real-world anomalies. Recent work integrates multimodal approaches, combining visual data with textual descriptions [42 , 58 , 62 , 65], while advances in multimodal visuallanguage models (VLMs) [6 , 21 , 27 , 29 , 68] have enabled more nuanced understanding through anomaly-related instruction tuning and text generation [9 , 36 , 47 , 67].

Despite these advancements, a significant gap remains in models' ability to comprehend anomalies across multiple temporal scales. For instance, while anomalies such as explosions or fights may be captured in a single frame, more complex events like theft or arson require understanding long-term contextual patterns. Existing VAU datasets [46 , 56] typically provide annotations at a single level of granularity, limiting models to understanding either immediate perceptual anomalies or those requiring extended contextual reasoning. The lack of datasets with hierarchical annotations—encompassing both shortterm and long-term anomalies—hinders models' capacity to reason about anomalies with diverse temporal characteristics. Moreover, constructing datasets that encapsulate this hierarchical complexity poses significant challenges in scalability and annotation quality.

To address these issues, we develop a semi-automated annotation engine that efficiently scales high-quality annotation by combining manual video segmentation with recursive free-text annotation using large language models (LLMs). The process involves three key stages: 1) hierarchical video decoupling, where we manually identify anomaly events and segment them into shorter clips; 2) hierarchical free-text annotation, where captions for each clip are generated through human effort or video captioning models, then summarized at the event and video levels via LLMs; and 3) hierarchical instruction construction, where the textual data is transformed into question-answer instruction prompts by combining captions and summaries with designed prompts, creating a dataset with rich annotations for training and evaluating models.

Utilizing the annotation engine, we introduce HIVAU70k, a large-scale video anomaly understanding benchmark with hierarchical instructions. Our dataset comprises over 70,000 multi-granular instruction data organized across clip-level, event-level, and video-level segments as shown in Fig. 1. This hierarchical structure empowers models to detect immediate anomalies, e.g., sudden explosions or fighting, as well as complex events that require an understanding of long-term context, like theft or arson. By annotating at multiple temporal levels, HIVAU-70k provides diverse anomalies in open-world scenarios.

Towards long-term VAU, efficiency remains a critical challenge. Previous methods [9 , 47 , 67] often rely on uniform frame sampling, which can either miss crucial anomaly frames or incur large computational costs [25 , 47 , 67]. To address this, we propose the Holmes-VAU method, which combines the proposed Anomaly-focused Temporal Sampler (ATS) with the multimodal visual-language model for efficient long-term video anomaly understanding (See Fig. 1). The ATS combines a anomaly scorer with a density-aware sampler that adaptively selects frames by their anomaly scores. This integration ensures that the visual-language model concentrates on anomaly-rich regions, enhancing both efficiency and accuracy.

Our contributions are threefold: 1) We introduce HIVAU-70k, a large-scale, multi-granular benchmark for hierarchical video anomaly understanding. 2) We propose the Holmes-VAU method, which combines the proposed Anomaly-focused Temporal Sampler (ATS) to boost the efficiency and accuracy of inference on long-term videos. 3) We conduct extensive experiments demonstrating the effectiveness of hierarchical instruction data in enhancing anomaly comprehension and validate the performance gains provided by the integrated ATS and visual-language model in processing long videos.

## 2. Related Works

Video Anomaly Detection. This task aims to temporally detect abnormal frames in a long untrimmed video [1 , 14 , 23 , 34 , 40 , 52]. Early attempts are based on handcrafted features [1 , 19 , 23 , 34 , 40 , 72]. Recently, deep learning approaches [14 , 37 , 61] have become dominant, broadly classified into unsupervised, weakly-supervised, and fully-supervised methods. Unsupervised methods train only on normal videos to learn normal patterns and are often designed as reconstruction-based [13 , 14 , 59 , 61], prediction-based [31], or a combination [33]. Some methods [48 , 50 , 66] also explore a fully unsupervised setting, including both normal and abnormal videos in training set without real labels. Weakly-supervised methods [11 , 22 , 37 , 46 , 49 , 55 , 56 , 70 , 73 , 74] use both normal and abnormal videos with video-level annotations. Fully-supervised methods [20 , 30] are less studied due to the high cost of precise frame-level annotations.

Multi-modal Video Anomaly Understanding. Largescale visual-language pre-trained models such as CLIP [44]

serve as a bridge between visual and textual modalities. Some recent works [17 , 42 , 58 , 62] in the realm of video anomaly detection have leveraged textual information as prompts to enhance the model's anomaly representation. Based on this, [57] firstly proposed the open vocabulary VAD task, [65] introduced a multimodal video anomaly dataset composed of dense clip captions. Furthermore, [67] extracted captions from video frames and designed prompts for LLMs to provide anomaly scores, [9] and [47] construct diverse and interactive instruction data at the video level. However, these datasets consider only a single temporal level of anomaly understanding data construction, i.e . clip-level [65] or video-level [9 , 47]. Unlike these methods, we focus on building large-scale hierarchical video anomaly understanding data for multimodal instruction tuning.

Hierarchical Video Understanding. Video understanding is a challenging task due to its temporal-scale diversity. To better comprehend videos, many previous works have focused on both datasets and models in hierarchical video understanding. For example, [3 , 24 , 32 , 35 , 45 , 54] proposed fine-grained action recognition and localization datasets, [16] provided free-form hierarchical captions for hour-long videos at multiple temporal scales, and [8 , 18 , 38 , 43 , 60 , 69] trained models on hierarchical levels to obtain better video feature representation. Recently, to assess the capability of video vision-language models (VLMs) in handling challenges in real-world scenarios, several video benchmarks [10 , 12] have been built, incorporating data at multiple temporal scales for evaluation. Unlike these works, we dive into the field of Video Anomaly Understanding, thus filling the gap in multi-scale annotations in this area.

## 3. HIVAU-70k Benchmark

We first define the video anomaly understanding task in Sec.3.1. Then, the construction process of the HIVAU-70k benchmark will be elaborated in Sec.3.2. Finally, we will present HIVAU-70k's statistical information in Sec. 3.3 .

## 3.1. Task Description

This work focuses on video anomaly understanding, involving temporal anomaly detection and anomaly explainability. Temporal anomaly detection aims to predict an anomaly score for each frame in a video V, represented as S ∈ R T , where T is the total number of frames. Building on this, we explore the model's ability to generate explanatory text outputs related to anomalies based on video input and user queries. Specifically, given a video V and a text query Q, the model generates an anomaly-related response A. We consider two key abilities: 1) visual perception, which involves recognizing main entities in the video, and 2) anomaly reasoning, which encompasses the

Figure 2. Data Engine. We present a structured workflow for generating hierarchical annotations across video, event, and clip levels. Clips are first captioned, then processed through a large language model (LLM) with prompts for event summarization. The outputs include clip captions, event summaries, and video summaries, followed by manual checking and refinement. This multi-step approach enriches the dataset with detailed judgments, descriptions, and analyses of anomalies, enabling robust contextual understanding at varying granularities.

![Image](artifacts/image_000001_10bd6193976d714ca1020d24629297cdd7ec97263edab23c1855a509ea4d278b.png)

model’s judgment and analysis of the anomaly content.

## 3.2. LLM-Empowered Data Engine

As shown in Fig.2, we develop a semi-automated annotation engine that efficiently scales high-quality annotations, which consists of three main steps: 1) hierarchical video decoupling, 2) hierarchical free-text annotation, and 3) hierarchical instruction construction.

Hierarchical Video Decoupling. Our video sources include the training set of the UCF-Crime [46] dataset and the XD-Violence [56] dataset, which contains videos of varying durations and diverse real-world anomalies. For abnormal videos, we first manually obtain the temporal boundaries of each anomaly event in the video. Non-continuous anomalous states are considered separate events. Then, we divide each event into clips of random lengths. For normal videos, we apply random sampling to obtain corresponding segments of varying granularities. Ultimately, we obtained 5,443 videos, 11,076 events, and 55,806 clips. This process took 5 annotators approximately 20 hours to complete. More details can be found in the appendix.

Hierarchical Free-text Annotation. To fully extract semantic information from the clip-level videos, we utilize a powerful off-the-shell video perception model LLaVANext-Video [28] to generate detailed captions for each clip. We also include the UCA dataset [64], which provides manually annotated captions for video clips in the UCFCrime [46] dataset. Then, we use an LLM [2] to consoli-

Figure 3. HIVAU-70k dataset. (a) Duration distributions for clips, events, and full videos, showing dominance of short clips. (b) Hierarchical data organization from clip-level to video-level, enabling perception-to-reasoning insights. (c) Word count variations across annotation levels, with more detailed descriptions at the video level. (d) Sample annotations capturing captioning, judgment, description, and anomaly analysis, highlighting nuanced understanding of anomaly events in complex scenes.

![Image](artifacts/image_000002_6fb6970c5cf463eb5e56670032057c5fb35d1f437baa8fe40a7c7f3a0525b937.png)

date all clip captions within an event, generating an eventlevel video summary. Specifically, we design prompts 1 to guide the LLM to produce three parts of content for each anomalous event summary: 1) Judgment: A determination of whether an anomaly exists and its specific category, 2) Description: A detailed description of the anomalous or normal event, 3) Analysis: The reasoning behind the anomaly judgment, including causal analysis. To guide the LLM in generating reliable responses, we also inject the event's category label (e.g., "Shooting", "Explosion") into the LLM prompt. We consolidate all event-level summaries to obtain the video-level summary. This results in free-text annotations across multiple temporal scales, including short-term visual perception (clip-level) and longterm anomaly reasoning (event-level, video-level).

Hierarchical Instruction Data Construction. The ability of VLMs to follow user instructions and generate responses is achieved through instruction-tuning [21 , 25 , 27 , 29]. The training data format typically consists of the following:

' {Q:user instruction , A:model response.} '

To build instruction-tuning data for VLMs in the domain of Video Anomaly Understanding, we matched free-text annotations with pre-designed anomaly-related user instructions. Specifically, for clip-level segments, we only construct instructions related to captions, as it is challenging to obtain anomaly-related analysis for short videos. For event-level and video-level segments, we construct instruction data from the perspectives of Judgment, Description, and Analysis. Typical examples are shown in the Fig. 3 (d). Manual Checking for Data Quality Control. To ensure dataset quality, we implemented several human inspec-

1 For detailed prompts, please refer to the appendix.

tion and curation strategies. First, we labeled the temporal boundaries of abnormal event segments and reviewed them at the second level. Next, anomaly labels were incorporated during summary generation using LLMs, directing focus on relevant entities. Finally, manual reviews were performed to correct low-quality instruction data.

## 3.3. Data Statistic of HIVAU-70k

Utilizing the proposed annotation engine, we introduce HIVAU-70k, as shown in Fig. 3, a large-scale benchmark designed for hierarchical instruction-based video anomaly understanding. As shown in Fig. 3(b), HIVAU-70k contains over 70,000 multi-granular annotations organized at clip-level, event-level, and video-level segments, achieving a progression from perception to reasoning. As shown in Fig. 3(a) and (c), the durations of segments and the word numbers of text annotations at different granularities exhibit significant distributional differences. As shown in Fig. 3(d), HIVAU-70k's instruction data covers Caption , Judgment , Description, and Analysis for real-world anomalies, which guide the model to develop both short-term and long-term video anomaly understanding capabilities.

## 4. Method: Holmes-VAU

Long-term video anomaly understanding with LLMs/VLMs has traditionally been hindered by frame redundancy, complicating accurate anomaly detection. Previous VAU approaches struggle with focus: methods like dense window sampling [67] add redundancy, and uniform frame sampling [9 , 47] often misses key anomalies, limiting application to short videos. We introduce the Anomaly-focused Temporal Sampler (ATS) to address this,

Figure 4. Holmes-VAU: a multi-modal-LLM-based video anomaly detection framework with adaptive anomaly focus.

![Image](artifacts/image_000003_35873d061bd20de07066d9e6a84bbeb000d94c06a88cf3d8012af4213a558c7a.png)

integrating it into the VLM, and fine-tuning it via instruction on HIVAU-70k to form our Holmes-VAU model.

## 4.1. Pipeline Overview

The overall pipeline of our Holmes-VAU model is shown in Fig. 4. Video frames are processed by a visual encoder, creating visual tokens. These tokens are analyzed by an Anomaly-focused Temporal Sampler using an anomaly scorer and cumulative sum (S cumsum ) to select keyframes. A text encoder processes a prompt (e.g., 'Describe the abnormal events in the video'). Visual and textual representations are combined in a pre-trained language model, fine-tuned with LoRA, to generate a description of detected anomalies, such as a 'collision between a car and a motorcycle' and 'traffic accident' indicators.

## 4.2. Model Architecture

Visual and Text embedding. We utilize the frozen visual encoder in InternVL2 [6], which inherits the ViT structure from CLIP [44], we refer to it as ϕ v . Following previous VAU works [55 , 58 , 67 , 74], we sample dense video frames at an interval of 16 frames from the input video. Each video frame is then processed through the visual encoder to obtain the corresponding visual tokens. Given the input video frame sequence V ∈ R T ×H×W×C , the output features of i-th frame can be denotes as:

<!-- formula-not-decoded -->

where v cls i indicates the class token, v j i (j ∈ {1 , 2, ..., Np Np }) denotes the patch tokens, and Np Np reperesents the number of patches. The text encoder ϕt is also initialized from [6], which includes a tokenizer and an embedding layer. The prompt text Q is converted into text tokens through the text encoder: X q = ϕt(Q) .

Anomaly-focused Temporal Sampler (ATS). ATS consists of two components: the anomaly scorer and the density-aware sampler. The anomaly scorer ϕ s is a featurebased VAD network which estimates the anomaly score for each frame. We follow the network architecture in [74] due to its simplicity and good performance. Given the class token of the video frames {v cls 1 , v cls 2 , ..., v cls T }, the anomaly scores can be obtained: s i = ϕ s (v cls i ), where si denotes the anomaly score of the i-th frame.

Anomalous frames typically contain more information and exhibit greater variation than normal frames [49]. This observation motivates us to sample more frames in regions with higher anomaly scores while reducing the sampling in areas with lower anomaly scores. As shown in Fig. 4, to achieve non-uniform sampling, we propose density-aware sampler to selectively choose N frames from a total of T input frames. Specifically, we treat the anomaly scores S ∈ R T as a probability mass function and first accumulate them along the temporal dimension to obtain the cumulative distribution function, denoted as S cumsum :

<!-- formula-not-decoded -->

We uniformly sample N points along the cumulative axis, then map these points to the cumulative distribution S cumsum , the corresponding N timestamps on the time axis are mapped to the closest frame index and finally form the sampled frame indices, denoted as G . τ is used to control the uniformity of the sampling.

Projector and LLM. We select the tokens corresponding to the sampled frame, i.e ., G, as the visual embedding. A projector ϕ p is then used to map the visual embedding to the language feature space. Finally, we concatenate these embeddings with the text embeddings, input them into the pre-trained large language model, and compute the probability of the target answers X a . To obtain an initial visuallanguage alignment space, we initialize the projector and LLM parameters from [6], with the parameters kept frozen. The above process can be expressed as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where cat[·] represents the concatenation operation, θ is the trainable parameters, Xins,&lt;i, Xa,&lt;i are the instruction and answer tokens in all turns before the current prediction token x i
, L is the length of sequence, respectively.

## 4.3. Training and Testing

Training. We train the model following two steps. In the first step, we use the video data and annotated framelevel label (yˆ ˆ ∈ R T ) from HIVAU-70k to train the anomaly scorer , which provides more accurate anomaly supervision compared to previous unsupervised and weakly-supervised methods [14 , 46 , 55 , 74].

<!-- formula-not-decoded -->

In the second step, we keep the anomaly scorer fixed, and use all the instruction data from HIVAU-70k to train the model. To achieve more efficient fine-tuning without disrupting the original capabilities of the LLM, we employ LoRA [15] for fine-tuning, optimizing the cross entropy loss between the predicted and the ground truth tokens.

Testing. During testing, the user inputs a video and text prompts; the model will generate the corresponding text response following the user's instruction.

## 5. Experiments

## 5.1. Experiment Setup

Dataset. Our HIVAU-70k is built upon two large-scale realworld datasets, i.e., UCF-Crime [46] and XD-Violence [56], they provide a diverse range of videos with anomalous events. UCF-Crime [46] comprises 1,900 untrimmed videos totaling 128 hours from outdoor and indoor surveillance cameras. It encompasses 13 classes of real-world anomalies, including Abuse , Explosion , Fighting, and Shooting. XD-Violence [56] is the largest VAD benchmark, comprising 4,754 videos totaling 217 hours sourced from surveillance, movies, car cameras, and games. It encompasses 6 anomaly classes: Abuse , Car Accidents , Explosions , Fighting , Riots, and Shooting .

Metric. We assess the anomaly understanding ability from two aspects: anomaly detection and reasoning . 1) For anomaly detection, we use the anomaly scores output by the Anomaly Scorer as the prediction and perform the evaluation. Following [14 , 46 , 67 , 74], we use AUC and AP to quantify detection performance, which is evaluated only on the video level. 2) For anomaly reasoning, we annotate instruction data from the UCF-Crime and XD-Violence test sets, which have been carefully reviewed and filtered by annotators. We finally collected 3,300 test samples at multiple granularities. The test set contains 2200/732/398 samples at clip/event/video levels. We calculate metrics including BLEU [41], CIDEr [51], METEOR [4] and ROUGE [26] to measure the quality of the reasoning text output by the model, comparing with the annotated ground truth text.

Implementation Details. For the proposed Holmes-VAU

Table 1. Comparison of detection performance with state-ofthe-art Video Anomaly Detection approaches. We include the results of explainable and non-explainable methods. "∗" represents the result reported in [67].

| Methods                     | Backbone                    | XD-Violence                 | UCF-Crime AUC/%            |
|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| Methods                     | Backbone                    | AP/%                        | AUC/%                       |
| Non-explainable VAD         | Non-explainable VAD         | Non-explainable VAD         | Non-explainable VAD         |
| Conv-AE [14] (CVPR’16)      | -                           | 27.25                       | 50.60                       |
| GODS [52] (ICCV’19)         | I3D                         | N/A                         | 70.46                       |
| GCL [66] (CVPR’22)          | ResNext                     | N/A                         | 71.04                       |
| DYANNET [48] (WACV’23)      | I3D                         | N/A                         | 84.50                       |
| MIST [11] (CVPR’21)         | I3D                         | N/A                         | 82.30                       |
| Wu et al. [56] (ECCV’20)    | I3D                         | 78.64                       | 82.44                       |
| RTFM [49] (ICCV’21)         | I3D                         | 77.81                       | 84.30                       |
| MSL [22] (AAAI’22)          | I3D                         | 78.28                       | 85.30                       |
| S3R [55] (ECCV’22)          | I3D                         | 80.26                       | 85.99                       |
| MGFN [5] (AAAI’23)          | I3D                         | 79.19                       | 86.98                       |
| UR-DMU [74] (AAAI’23)       | I3D                         | 81.66                       | 86.97                       |
| CLIP-TSA [17] (ICIP’23)     | ViT                         | 82.19                       | 87.58                       |
| VadCLIP [58] (AAAI’24)      | ViT                         | 84.51                       | 88.02                       |
| Yang et al. [62] (CVPR’24)  | ViT                         | 83.68                       | 87.79                       |
| Wu et al. [57] (CVPR’24)    | ViT                         | 66.53                       | 86.40                       |
| Explainable Multi-modal VAD | Explainable Multi-modal VAD | Explainable Multi-modal VAD | Explainable Multi-modal VAD |
| Zero-Shot CLIP [44] ∗      | ViT                         | 17.83                       | 53.16                       |
| LLAVA-1.5 [27] ∗           | ViT                         | 50.26                       | 72.84                       |
| LAVAD [67] (CVPR’24)        | ViT                         | 62.01                       | 80.28                       |
| Holmes-VAU (Ours)           | ViT                         | 87.68                       | 88.96                       |

method, we initialize the Multimodal LLM with InternVL22B [6]. To optimize the Anomaly-focused Temporal Sampler, we adopt the Adam optimizer with a learning rate of 1e-4. Note that when evaluating detection performance on XD-Violence and UCF-Crime, only videos in the corresponding training sets are used to train our model for fair comparisons. For instruction tuning, we train with a batch size of 512 for 1 epoch, using the AdamW optimizer with cosine learning rate decay and a warm-up period. The LoRA [15] parameters are set as: r=64, α=128, and learning rate=4e-5. During testing, τ in Eq. 2 is set to 0.1. Experiments are conducted on 2 NVIDIA A100 GPUs.

## 5.2. Main Results

Anomaly Detection Results. We compare our method with state-of-the-art methods, including semi-supervised methods [14 , 52], unsupervised methods [48 , 66], weaklysupervised methods [17 , 22 , 49 , 55 , 58 , 74] and recently training-free method [67]. We have indicated their backbones and performance on the UCF-Crime and XDViolence datasets, as shown in Table 1. Our method has an AP of 87.68% on XD-Violence and an AUC of 88.96% on UCF-Crime, significantly outperforming the prior state-ofthe-art methods, which demonstrates that our method can generate less biased anomaly scores. It is worth noting that while achieving precise localization of anomalies, HolmesVAU is also capable of providing explanations and analysis for the detected anomalies by the model, a feature unavailable in existing non-explainable VAD methods. Although LAVAD [67] has explainability, the training-free large lan-

Table 2. Comparison of reasoning performance with state-of-the-art Multimodal Large Language Models (MLLMs). 'BLEU' refers to the cumulative values from BLEU-1 to BLEU-4. We evaluate the quality of the generated text at different granularities, including cliplevel (C), event-level (E), and video-level (V).

| Method                | Params    | BLEU [41](↑) C   | BLEU [41](↑) C   | BLEU [41](↑) C   | CIDEr [51](↑)    | CIDEr [51](↑)    | CIDEr [51](↑)    | METEOR [4](↑)    | METEOR [4](↑)    | METEOR [4](↑)    | ROUGE [26](↑) C E V   | ROUGE [26](↑) C E V   | ROUGE [26](↑) C E V   |
|-----------------------|-----------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------------|------------------------|------------------------|
| Method                | Params    | C                | E                | V                | C                | E                | V                | C                | E                | V                | C                      | E                      | V                      |
| Video-ChatGPT [39]    | 7B        | 0.152            | 0.068            | 0.066            | 0.033            | 0.011            | 0.013            | 0.102            | 0.069            | 0.044            | 0.153                  | 0.048                  | 0.079                  |
| Video-LLaMA [68]      | 7B        | 0.151            | 0.079            | 0.104            | 0.024            | 0.014            | 0.017            | 0.112            | 0.076            | 0.057            | 0.156                  | 0.067                  | 0.090                  |
| Video-LLaVA [25]      | 7B        | 0.164            | 0.046            | 0.055            | 0.032            | 0.009            | 0.013            | 0.097            | 0.022            | 0.014            | 0.132                  | 0.023                  | 0.045                  |
| LLaVA-Next-Video [71] | 7B        | 0.435            | 0.091            | 0.120            | 0.102            | 0.015            | 0.031            | 0.117            | 0.085            | 0.096            | 0.198                  | 0.080                  | 0.106                  |
| QwenVL2 [53]          | 7B        | 0.312            | 0.082            | 0.155            | 0.044            | 0.020            | 0.044            | 0.133            | 0.092            | 0.112            | 0.163                  | 0.081                  | 0.137                  |
| InternVL2 [6]         | 8B        | 0.331            | 0.101            | 0.145            | 0.052            | 0.022            | 0.035            | 0.141            | 0.095            | 0.101            | 0.182                  | 0.102                  | 0.122                  |
| Holmes-VAU (Ours)     | 2B        | 0.913            | 0.804            | 0.566            | 0.467            | 1.519            | 1.437            | 0.190            | 0.165            | 0.121            | 0.329                  | 0.370                  | 0.355                  |

Table 3. Ablation of hierarchical instruction data. During the instruction tuning phase, we combined training data of different granularities, including clip (C), event (E), and video (V) levels.

| Training Data    | Training Data    | Training Data    | BLEU(↑)    | BLEU(↑)    | BLEU(↑)    | CIDEr(↑)   | CIDEr(↑)   | CIDEr(↑)   |
|------------------|------------------|------------------|------------|------------|------------|------------|------------|------------|
| C                | E                | V                | C          | E          | V          | C          | E          | V          |
| ✓                |                  |                  | 0.984      | 0.261      | 0.351      | 0.459      | 0.120      | 0.106      |
|                  | ✓                |                  | 0.508      | 0.576      | 0.292      | 0.097      | 1.183      | 0.872      |
|                  |                  | ✓                | 0.280      | 0.222      | 0.279      | 0.039      | 0.708      | 0.884      |
| ✓                | ✓                |                  | 0.889      | 0.741      | 0.349      | 0.470      | 1.285      | 0.889      |
| ✓                |                  | ✓                | 0.906      | 0.341      | 0.522      | 0.472      | 0.962      | 1.093      |
| ✓                |                  | ✓                | 0.394      | 0.797      | 0.505      | 0.081      | 1.472      | 1.074      |
| ✓                | ✓                | ✓                | 0.913      | 0.804      | 0.566      | 0.467      | 1.519      | 1.437      |

guage model lacks an understanding of anomaly knowledge due to the limitation of insufficient supervised data.

Anomaly Reasoning Results. We compare the anomalyrelated text quality generated by Holmes-VAU with that produced by state-of-the-art general Multimodal Large Language Models (MLLMs), and presented the results at different temporal granularities, including clip-level, event-level, and video-level, as shown in Table. 2. Earlier MLLMs such as Video-ChatGPT [21] and Video-LLaMA [68], struggled with basic visual perception and instruction-following capabilities. Recent MLLMs [6 , 7 , 71] trained on larger and higher-quality video instruction data have made significant progress in general video understanding, with noticeable improvements at the clip-level perception task. However, due to the absence of learning from complex, real-world anomaly data, their reasoning abilities at the event-level and video-level are still lacking. Our Holmes-VAU, however, shows significant improvements in video understanding across all temporal granularities compared to existing general MLLMs, highlighting the importance of injecting anomaly-related knowledge through instruction tuning on high-quality Video Anomaly Understanding benchmarks.

## 5.3. Analytic Results

Influence of Hierarchical Instruction. To explore the impact of different granularity video training data on the model's anomaly reasoning ability, we designed various training data combinations during the instruction tuning phase and evaluated the model's performance, as shown

Table 4. Ablation study of sampling methods and the number of sampled frames. We compare the proposed Anomaly-focused Temporal Sampler (ATS) with other sampling methods under different frame sampling numbers, including Uniform and Top-K. Latency is the time delay in generating the first token.

| Frames (N)    | Sampler                            | Latency (ms)    | Video-level                  | Video-level        |
|---------------|------------------------------------|-----------------|------------------------------|--------------------|
| Frames (N)    | Sampler                            | Latency (ms)    | BLEU(↑)                      | CIDEr(↑)           |
| 8             | Top-K  Uniform  ATS (Our         | 244             | 0.462  0.491  0.514  0476 | 1.229 1.276 1324 |
| 16            | Uniform  ATS (Ours)  Top-K  Uif | 244             | 0.514                        | 1.324              |
| 16            | Top-K  Uniform  ATS (Our         | 566             | 0.476  0.511  0566         | 1.302 1.345 1437 |
| 16            | TopK  Uniform  ATS (Ours         | 566             | 0.511                        | 1.345              |
|               | Uf ATS (Ours)  Top-K             | 1402            | 0.566  0.481  0558         | 1.437 1.332       |

in Table 3. The inclusion of clip-level data primarily enhanced the model's basic visual perception abilities regarding actions and scenes within the video. Adding eventlevel data improved the model's ability to judge and understand complete anomaly events. Furthermore, the involvement of video-level data further enhanced the model's ability to analyze and summarize anomaly-related information across longer-span videos. The hierarchical instruction data structure facilitated a comprehensive and complementary improvement in the model's anomaly-related perception-toreasoning capabilities.

Influence of different sampling methods and the number of sampled frames. Our ATS (Anomaly-focused Temporal Sampler) is designed to adaptively sample frames input to the LLM based on the frame-level anomaly scores. To validate its advantages, we compared ATS with other sampling methods at various sample frame counts, including Uniform and Top-K sampling. In Uniform sampling, N frames are uniformly sampled from all frames, while Top-K sampling selects the frames with the top N highest anomaly scores. As shown in Table 4, ATS consistently outperforms other sampling methods, regardless of the sample count. We believe that Uniform sampling tends to overlook key anomaly frames, though this issue lessens as more frames are sampled. Besides, Top-K sampling tends to overly focus on local anomaly frames, missing contextual frame informa-

Figure 5. Qualitative comparison of anomaly understanding explanation. Compared with state-of-the-art general MLLMs, i.e., InternVL2 [6] and QwenVL2 [53], our proposed Holmes-VAU demonstrates more accurate anomaly judgment, along with more detailed and comprehensive anomaly-related descriptions and analysis. Correct and wrong explanations are highlighted in green and red, respectively.

![Image](artifacts/image_000004_1a2437cc42c3e51dad22263d510ac6007e27e642009729675e1b511421b35d3a.png)

Figure 6. Ablation study of trainable parameters. (a) Loss curve during instruction-tuning. (b) We tuned the LoRA [15] parameter r to control trainable parameters, evaluating its impact on VAU capability, and General performance on Video-MME [12].

![Image](artifacts/image_000005_6151cc2077c0ba8602216d0ae3a10ef0a2c360f7c179df7e204c7af1bef792d1.png)

tion. Our proposed ATS mitigates both issues. To balance inference efficiency and performance, we set N=16 as the default sample frame number.

Instruction Tuning Parameters. We conducted an ablation study on the parameter r in LoRA [15] to explore how the trainable parameters affects both the model's VAU performance and its general capability. We use VideoMME [12] to evaluate the model's general capability. The results are shown in Fig. 6, as r increases, the model gradually adapts to the VAU task. However, when r becomes too large, the model's general capability decreases. To retain the original general video understanding capability of the MLLM, we set r=64 as the default value.

## 5.4. Qualitative Comparision

We provide qualitative comparisons between Holmes-VAU and existing MLLMs in Fig. 5. The results demonstrate that Holmes-VAU can accurately identify anomalies in videos and provide accurate and complete explanations, highlight the effectiveness and advantage of Holmes-VAU in perceiving video events and analyzing anomalies.

## 6. Conclusion

In conclusion, this work pushes the boundaries of video anomaly understanding by introducing hierarchical anomaly detection across diverse temporal scales, from momentary clips to extended events. The HIVAU-70k benchmark, with over 70,000 multi-level annotations, addresses a critical gap in the field, enabling comprehensive anomaly analysis in real-world scenarios. Our Anomaly-focused Temporal Sampler (ATS) strategically enhances focus on anomaly-dense segments, optimizing both efficiency and accuracy in long-term anomaly detection. Extensive experiments demonstrate that our hierarchical dataset and ATS-enhanced VLM achieve significant performance gains over conventional methods, proving robust for open-world anomaly understanding. This work sets a new standard for multi-granular anomaly comprehension, paving the way for more fine-grained video anomaly understanding.

Acknowledgement This work is supported by the National Natural Science Foundation of China under grants U22B2053 and 623B2039, and in part by the Interdisciplinary Research Program of HUST (2024JCYJ034).

## References

- [1] Amit Adam, Ehud Rivlin, Ilan Shimshoni, and Daviv Reinitz. Robust real-time unusual event detection using multiple fixed-location monitors. IEEE transactions on pattern analysis and machine intelligence, 30(3):555–560, 2008. 2
- [2] AI@Meta. Llama 3 model card. 2024. 3 , 13
- [3] Kumar Ashutosh, Rohit Girdhar, Lorenzo Torresani, and Kristen Grauman. Hiervl: Learning hierarchical videolanguage embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 23066–23078, 2023. 3
- [4] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005. 6 , 7
- [5] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 387–395, 2023. 6
- [6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2 , 5 , 6 , 7 , 8 , 17
- [7] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 7
- [8] Jisheng Dang, Huicheng Zheng, Xiaohao Xu, Longguang Wang, Qingyong Hu, and Yulan Guo. Adaptive sparse memory networks for efficient and robust video object segmentation. IEEE Transactions on Neural Networks and Learning Systems, 2024. 3
- [9] Hang Du, Sicheng Zhang, Binzhu Xie, Guoshun Nan, Jiayang Zhang, Junrui Xu, Hangyu Liu, Sicong Leng, Jiangming Liu, Hehe Fan, et al. Uncovering what why and how: A comprehensive benchmark for causation understanding of video anomaly. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18793– 18803, 2024. 2 , 3 , 4 , 16 , 17
- [10] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024. 3
- [11] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In Proceedings of the IEEE/CVF conference
12. on computer vision and pattern recognition, pages 14009– 14018, 2021. 2 , 6
- [12] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 3 , 8
- [13] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1705–1714, 2019. 2
- [14] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 733–742, 2016. 2 , 6
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 6 , 8
- [16] Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and Gedas Bertasius. Video recap: Recursive captioning of hour-long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18198–18208, 2024. 3
- [17] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In 2023 IEEE International Conference on Image Processing (ICIP), pages 3230–3234. IEEE, 2023. 3 , 6
- [18] Kumara Kahatapitiya and Michael S Ryoo. Coarse-fine networks for temporal activity detection in videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8385–8394, 2021. 3
- [19] Jaechul Kim and Kristen Grauman. Observe locally, infer globally: a space-time mrf for detecting abnormal activities with incremental updates. In 2009 IEEE conference on computer vision and pattern recognition, pages 2921–2928. IEEE, 2009. 2
- [20] Federico Landi, Cees GM Snoek, and Rita Cucchiara. Anomaly locality in video surveillance. arXiv preprint arXiv:1901.10364, 2019. 2
- [21] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2 , 4 , 7
- [22] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1395–1403, 2022. 2 , 6
- [23] Weixin Li, Vijay Mahadevan, and Nuno Vasconcelos. Anomaly detection and localization in crowded scenes. IEEE transactions on pattern analysis and machine intelligence , 36(1):18–32, 2013. 2

- [24] Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In Proceedings of the European Conference on Computer Vision (ECCV), pages 513–528, 2018. 3
- [25] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 2 , 4 , 7
- [26] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 6 , 7
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 2 , 4 , 6
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2 , 4
- [30] Kun Liu and Huadong Ma. Exploring background-bias for anomaly detection in surveillance videos. In Proceedings of the 27th ACM International Conference on Multimedia , pages 1490–1499, 2019. 2
- [31] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6536–6545, 2018. 2
- [32] Yi Liu, Limin Wang, Yali Wang, Xiao Ma, and Yu Qiao. Fineaction: A fine-grained video dataset for temporal action localization. IEEE transactions on image processing, 31: 6937–6950, 2022. 3
- [33] Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13588–13597, 2021. 2
- [34] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In Proceedings of the IEEE international conference on computer vision, pages 2720–2727, 2013. 2
- [35] Jian Lu, Xuanfeng Li, Bo Zhao, and Jian Zhou. A review of skeleton-based human action recognition. Journal of Image and Graphics, 28(12):3651–3669, 2023. 3
- [36] Hui Lv and Qianru Sun. Video anomaly detection and explanation via large language models. arXiv preprint arXiv:2401.05702, 2024. 2 , 16
- [37] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8022–8031, 2023. 2
- [38] Baiteng Ma, Shiwei Zhang, Changxin Gao, and Nong Sang. Temporal global correlation network for end-to-end action proposal generation. Acta Electronica Sinica, 50(10):2452– 2461, 2022. 3
- [39] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 7
- [40] Ramin Mehran, Alexis Oyama, and Mubarak Shah. Abnormal crowd behavior detection using social force model. In 2009 IEEE conference on computer vision and pattern recognition, pages 935–942. IEEE, 2009. 2
- [41] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002. 6 , 7
- [42] Yujiang Pu, Xiaoyu Wu, Lulu Yang, and Shengjin Wang. Learning prompt-enhanced context features for weaklysupervised video anomaly detection. IEEE Transactions on Image Processing, 2024. 2 , 3
- [43] Zhiwu Qing, Shiwei Zhang, Ziyuan Huang, Yi Xu, Xiang Wang, Mingqian Tang, Changxin Gao, Rong Jin, and Nong Sang. Learning from untrimmed videos: Self-supervised video representation learning with hierarchical consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13821–13831, 2022. 3
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2 , 5 , 6
- [45] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2616–2625, 2020. 3
- [46] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6479–6488, 2018. 2 , 3 , 6 , 13
- [47] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and YingCong Chen. Hawk: Learning to understand open-world video anomalies. arXiv preprint arXiv:2405.16886, 2024. 2 , 3 , 4 , 16 , 17
- [48] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. Dyannet: A scene dynamicity guided self-trained video anomaly detection network. In Proceedings of the IEEE/CVF Winter conference on applications of computer vision, pages 5541–5550, 2023. 2 , 6
- [49] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4975–4986, 2021. 2 , 5 , 6

- [50] Anil Osman Tur, Nicola Dall'Asen, Cigdem Beyan, and Elisa Ricci. Exploring diffusion models for unsupervised video anomaly detection. In 2023 IEEE International Conference on Image Processing (ICIP), pages 2540–2544. IEEE, 2023. 2
- [51] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015. 6 , 7
- [52] Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8201–8211, 2019. 2 , 6
- [53] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 7 , 8
- [54] SC Wang, Q Huang, YF Zhang, X Li, YQ Nie, and GC Luo. Review of action recognition based on multimodal data. Image Graph, 27(11):3139–3159, 2022. 3
- [55] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Self-supervised sparse representation for video anomaly detection. In European Conference on Computer Vision, pages 729–745. Springer, 2022. 2 , 5 , 6
- [56] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 322–339. Springer, 2020. 2 , 3 , 6 , 13
- [57] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18297–18307, 2024. 2 , 3 , 6
- [58] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6074–6082, 2024. 2 , 3 , 5 , 6
- [59] Dan Xu, Yan Yan, Elisa Ricci, and Nicu Sebe. Detecting anomalous events in videos by learning deep representations of appearance and motion. Computer Vision and Image Understanding, 156:117–127, 2017. 2
- [60] Xiaohao Xu, Jinglu Wang, Xiang Ming, and Yan Lu. Towards robust video object segmentation with adaptive object calibration. In Proceedings of the 30th ACM International Conference on Multimedia, pages 2709–2718, 2022. 3
- [61] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14592–14601, 2023. 2
- [62] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18899–18908, 2024. 2 , 3 , 6
- [63] Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David J Crandall. Dota: Unsupervised detection of traffic anomaly in driving videos. IEEE transactions on pattern analysis and machine intelligence, 45(1): 444–459, 2022. 2
- [64] Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance video-and-language understanding: New dataset, baselines, and challenges, 2023. 3 , 16
- [65] Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance video-and-language understanding: New dataset baselines and challenges. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22052– 22061, 2024. 2 , 3 , 13 , 17
- [66] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14744–14754, 2022. 2 , 6
- [67] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18527–18536, 2024. 2 , 3 , 4 , 5 , 6 , 16
- [68] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 2 , 7
- [69] Huaxin Zhang, Xiang Wang, Xiaohao Xu, Zhiwu Qing, Changxin Gao, and Nong Sang. Hr-pro: Point-supervised temporal action localization via hierarchical reliability propagation. arXiv preprint arXiv:2308.12608, 2023. 3
- [70] Huaxin Zhang, Xiang Wang, Xiaohao Xu, Xiaonan Huang, Chuchu Han, Yuehuan Wang, Changxin Gao, Shanjun Zhang, and Nong Sang. Glancevad: Exploring glance supervision for label-efficient video anomaly detection. arXiv preprint arXiv:2403.06154, 2024. 2
- [71] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 7 , 13
- [72] Bin Zhao, Li Fei-Fei, and Eric P Xing. Online detection of unusual events in videos via dynamic sparse coding. In CVPR 2011, pages 3313–3320. IEEE, 2011. 2
- [73] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1237–1246, 2019. 2
- [74] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video

anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3769–3777, 2023. 2 , 5 , 6 , 14 , 15

## Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

Supplementary Material

## A. Details of the Data Engine.

Brief explanation of the basis for judging the anomaly.’

To construct a dataset with hierarchical annotations with both short-term and long-term anomalies, we developed a semi-automated annotation engine that combines manual efforts with the generative capabilities of LLM. In the main paper, we present the complete annotation workflow. Below, we provide additional details about the data engine.

## A.1. Hierarchical Video Decoupling

Before annotation, we collected videos from the training sets of the UCF-Crime [46] and XD-Violence [56] datasets. From UCF-Crime, we selected 758 normal videos and 735 anomaly videos, while from XD-Violence, we selected 1,904 normal videos and 2,046 anomaly videos. The anomaly videos included their original video-level labels, e.g ., Abuse , Explosion. For the anomaly videos, we organized a team of five annotators to label each anomaly event within the videos. The annotation process took approximately 20 hours to complete. For the normal videos, we considered all segments to be normal and randomly cropped segments of varying lengths to serve as normal event-level video segments. These anomaly and normal event-level videos were further divided into shorter clip-level segments. For UCF-Crime, we adopted the clip-level divisions from UCA [65]. For XD-Violence, we performed uniform division.

## A.2. Hierarchical Free-text Annotation

Clip Captioning. For videos in UCF-Crime, we fully utilized the manually annotated captions from UCA [65]. For videos in XD-Violence, we used LLaVA-Next-Video7B [71] as our captioner to generate textual descriptions for clip-level videos. The specific prompt is as follows:

'Please provide a short and brief description of the video clip, focusing on the main subjects and their actions.'

Event Summary. We combined all captions and videolevel category labels to generate anomaly-related summaries for each event using an LLM. We selected LLaMA370B [2] as our LLM due to its strong summarization capabilities. The specific prompt is as follows:

'The dense caption of the video is: {clip captions} . There are (is no) abnormal events ({video-level label}) in the video. Your response should include the following three parts: 1. Whether the anomaly exists and the specific name of the anomaly. 2. A summary of the anomaly events. 3.

Video Summary. Similar to generating event summaries, we generated video-level summaries by analyzing the event-level summaries. The specific prompt is as follows:

'Below is a summary of all the events in the video: {event summaries}. There are (is no) abnormal events ({videolevel label}) in the video. Your response should include the following three parts: 1. Whether the anomaly exists and the specific name of the anomaly. 2. Detailed description of the video anomaly event from start to end. 3. Brief analysis of the basis for judging the anomaly.'

Annotation Format. In Fig.A, we present an example of the hierarchical free-text annotations for a video.

## A.3. Hierarchical Instruction Data Construction

To construct the instruction dataset, we designed question prompts tailored to different tasks, including Caption , Judgment , Description, and Analysis. For each instruction item, we randomly selected one prompt from the pool and matched it with the corresponding content from the free-text annotations as the answer.

## Caption.

1. "Describe the video briefly."
2. "Describe the main events that take place in this video."
3. "Give a short description of the video."
4. "What happened in this video?"
5. "Generate a brief caption for the video."

6.

”Can you provide a brief description of the video?”

7. "Briefly describe the main subjects and their actions in the video."
8. "Provide a short overview of what happens in the video?"
9. "Describe the key moments that showcase the subjects' activities in the video."
10. "Describe the sequence of events involving the main subjects in the video."
11. "What activities happen throughout the video?"
12. "Describe the main subjects and their roles in the video."
7. 13.
8. ”What key moments stand out in the video?”
14. "What are the primary activities showcased in the video?"
15. "What happens to the main subjects as the video progresses?"
16. "What is a brief overview of what happens in the video?"
17. "Describe the main subjects and their contributions to the video."
18. "Describe the key events in the video."
19. "Describe the video's main activities."
20. "Can you describe the main action in this video briefly?"
21. "Describe the video clip concisely."
22. "Provide a brief description of the given video clip."
23. "Summarize the visual content of the video clip."
24. "Give a short and clear explanation of the subsequent video clip."

## Judgement.

1. "What types of anomalies are shown in the video clip?"
2. "Are there any anomaly events detected in the video?"
3. "Detect and classify the anomaly events in the video."
4. "Identify any abnormal behaviors depicted in the video."
5. "Determine whether there are anomaly events in the video and the specific name of the anomaly."
6. "What anomalies can be identified in the video?"
7. "What categories of anomalies can be found in the video?"
8. "Could you point out any abnormal actions in the video?"
9. "Point out the abnormal actions in the video."

![Image](artifacts/image_000006_f12f7866864c2f8a349bb696b0762fc6a1d0658902d24ff259302dbe69fa5af5.png)

```
1 { 2 "video": "v=2rfyeR-YaJw__#1_label_G-0-0" , 3 "n_frames": 1940 , 4 "fps": 24.0 , 5 "label": ["Explosion"], 6 "clips": [[[5.583 , 11.903], [11.903 , 18.222], [18.222 , 24.542]], 7 [[36.167 , 43.48], [43.48 , 50.792]]], 8 "clip_captions":[ 9 [ 10 "A military tank moving across a barren landscape with low-rise buildings and sparse vegetation. the sky , → is overcast, and the overall color palette is muted with earthy tones." , 11 "A series of images depicting a barren landscape with a few buildings in the background. the foreground consists of a rocky terrain with sparse vegetation. the sky is overcast, and there are no visible people or moving objects." , , → , → 12 "A silhouette of a person operating a large, mounted weapon on a rocky terrain under a clear sky. the , → individual appears to be adjusting or aiming the weapon." 13 ], 14 [ 15 "A series of images depicting a hazy, dusty environment with buildings in the background. the focus is on a red light source that appears to be moving or flashing, possibly indicating some form of activity or event occurring in the area." , , → , → 16 "A series of images depicting a military tank moving across a landscape with buildings in the background. , → the tank is seen from different angles, and there are visible explosions occurring in the distance." 17 ] 18 ], 19 "events": [[5.583 , 24.542], [36.167 , 50.792]], 20 "event_summary": [ 21 "The anomaly exists, specifically an Explosion anomaly. The basis for judging the anomaly is the presence of an unexpected and sudden release of energy, likely accompanied by a loud noise and visible effects such as smoke, fire, or debris." , , → , → 22 "The anomaly exists and its specific name is Explosion. The anomaly event is a series of visible explosions occurring in the distance as a military tank moves across a landscape with buildings in the background. The basis for judging the anomaly is the sudden and intense release of energy and light in the form of explosions, which is an unusual and notable event in the otherwise desolate and quiet war-torn environment depicted in the video"], , → , → , → , → 23 "video_summary": "The anomaly exists, specifically named as Explosion. The anomaly event depicts a series of visible explosions occurring in the distance as a military tank moves across a landscape with buildings in the background, characterized by sudden and intense releases of energy and light, accompanied by loud noise and visible effects such as smoke, fire, or debris. The basis for judging the anomaly lies in the unusual and notable nature of these explosions, which stand out against the otherwise desolate and quiet war-torn environment depicted in the video, making them an unexpected and sudden release of energy that grabs attention." , → , → , → , → , → , → 24 }
```

Figure A. An example of hierarchical free-text annotations. For each labeled video, the hierarchical free-text annotations include cliplevel captions, event-level, and video-level anomaly analysis. Additionally, the temporal boundaries for each event and clip are annotated.

10. "Are there anomalies observed in the video clip?"

## Description.

1. "Describe the anomaly events observed in the video."
2. "Could you describe the anomaly events observed in the video?"
3. "Could you specify the anomaly events present in the video?"
4. "Give a description of the detected anomaly events in this video."
5. "Could you give a description of the anomaly events in the video?"
6. "Provide a summary of the anomaly events in the video."
7. "Could you provide a summary of the anomaly events in this video?""
8. "What details can you provide about the anomaly in the video?"
9. "How would you detail the anomaly events found in the video?"
10. "How would you describe the particular anomaly events in the video?"

## Analysis.

1. "Why do you judge this event to be anomalous?"
2. "Can you provide the reasons for considering it anomalous?"
3. "Can you give the basis for your judgment of this event as an anomaly?"
4. "What led you to classify this event as an anomaly?"
5. "Could you provide the reasons for considering this event as abnormal?"
6. "What evidence do you have to support your judgment of this event as an anomaly?"
7. "Can you analyze the factors contributing to this anomalous event?"
8. "Could you share your analysis of the anomalous event?"
9. "What patterns did you observe that contributed to your conclusion about this event being an anomaly?"
10. "How do the characteristics of this event support its classification as an anomaly?"

## A.4. Data Samples.

To facilitate understanding, we provide the final constructed instruction data at various temporal granularities, including clip-level , event-level, and video-level, as shown in Fig. B , Fig. C, and Fig. D .

## B. Details of the Anomaly Scorer

## B.1. Model Architecture

We use UR-DMU [74] as the anomaly scorer in our Anomaly-focused Temporal Sampler. As shown in Fig. E , UR-DMU utilizes a Global and Local Multi-Head Self Attention (GL-MHSA) module to capture both long-range and short-range temporal relationships among video snippets. Furthermore, UR-DMU introduces two memory banks to store and differentiate abnormal and normal prototypes, thereby maximizing the margins between these two representations. In order to learn discriminative representations, UR-DMU uses triplet loss to increase the feature distance

![Image](artifacts/image_000007_46f20ba091d40e7f83588f546a61af43ae983564f95de59f8aa5923afb71d733.png)

1

2

3

4

5

```
1 { 2 "id": "Abuse007_x264_E0C2" , 3 "type": "clip" , 4 "video": , → "ucf-crime/clips/train/Abuse007_x264_E0C2.mp4" , 5 "conversations": [ 6 {"from": "human" , 7 "value": "Describe the key events in the video."}, 8 {"from": "gpt" , 9 "value": "The man in shorts was slapped to the , → ground by the strong man opposite."} 10 ] 11 } 6 7 8 9 10 11 12 13 14
```

Figure B. An example of clip-level instruction data.

```
1 { 2 "id": "Abuse018_x264_E0" , 3 "type": "event" , 4 "video": , → "ucf-crime/events/train/Abuse018_x264_E0.mp4" , 5 "conversations": [ 6 {"from": "human" , 7 "value": "Identify any abnormal behaviors depicted , → in the video."}, 8 {"from": "gpt" , 9 "value": "An anomaly exists, specifically an , → instance of Police Brutality."}, 10 {"from": "human" , 11 "value": "Could you provide a summary of the anomaly , → events in this video?"}, 12 {"from": "gpt" , 13 "value": "The anomaly event involves the police officer kicking the man who rushed towards him, causing the man to fall heavily to the ground with his head on his back."}, , → , → , → 14 {"from": "human" , 15 "value": "Could you share your analysis of the , → anomalous event?"}, 16 {"from": "gpt" , 17 "value": "The excessive use of force by the police officer, which deviates from standard protocol and may be considered an abuse of authority."} , → , → 18 ] 19 }
```

Figure C. An example of event-level instruction data.

after interacting with different memories. Simultaneously, it utilizes KL loss to constrain the normal memory to follow a Gaussian distribution, accounting for the variance introduced by noise. Furthermore, We leveraged the event-level anomaly boundaries obtained during the annotation phase to generate frame-level labels and computed a binary crossentropy loss, i.e ., LAS, which is simple yet effective. Thus, the loss function for the anomaly scorer is defined as follows:

<!-- formula-not-decoded -->

15

16

17

18

19

20

```
{ "id": "v=xLFbqPzSI2Y__#1_label_B4-0-0" , "type": "video" , "video": "xd-violence/videos/train/ v=xLFbqPzSI2Y__#1_label_B4-0-0.mp4" , "conversations": [ {"from": "human" , "value": "Are there any anomaly events detected in , → the video?"}, {"from": "gpt" , "value": "The anomaly exists and its specific name , → is the Riot anomaly."}, {"from": "human" , "value": "Could you give a description of the , → anomaly events in the video?"}, {"from": "gpt" , "value": "The anomaly event depicts a chaotic and intense urban unrest scene, starting with a chaotic street scene where individuals are running away from an area of fire and smoke, with people appearing to be in a state of panic or urgency, and some being sprayed with water by individuals in uniform. The scene escalates to show a growing fire in the background, police officers in riot gear standing on the street, and a formation of police officers spread out across the road, suggesting an ongoing riot or civil unrest. "}, , → , → , → , → , → , → , → , → , → , → , → {"from": "human" , "value": "Can you give the basis for your , → judgment?"}, {"from": "gpt" , "value": "The presence of multiple indicators of a riot or emergency situation, including fire, smoke, and people fleeing in panic, as well as the sudden appearance of flames, smoke, and police in riot gear, and the presence of violent and destructive events, such as fires and explosions, and the response of authorities, including the use of tear gas or pepper spray, which deviate significantly from normal, peaceful urban activity."} , → , → , → , → , → , → , → , → , → ] }
```

Figure D. An example of video-level instruction data.

Figure E. Architecture of the Anomaly Scorer (UR-DMU [74]).

![Image](artifacts/image_000008_5d842bb2caa686e84c4403f1c6398884a38445827cd801427926e4d3480a8722.png)

## B.2. Visualization Results

In Fig. F, we present visualized results of anomaly scores and sampled frames on the UCF-Crime and XD-Violence test sets. These results demonstrate the accuracy of our

Figure F. Visualization results of anomaly scores and sampled frames output by the Anomaly-focused Temporal Sampler. .

![Image](artifacts/image_000009_c495b0d18021e42153eb9af079800455fca2799ce8b19aba0b041999750e56d3.png)

Table A. Comparision with related multimodal/explanable VAU methods and benchmarks. HIVAU-70k provides accurate temporal annotations and hierarchical anomaly-related free-text annotations.

| Methods             | #Catogories    | #Samples    | Text       | Text        | Text        | Temp. Anno.    | MLLM tuning   |
|---------------------|----------------|-------------|------------|-------------|-------------|----------------|---------------|
|                     |                |             | clip-level | event-level | video-level |                |               |
| UCA [64]            | 13             | 23,542      | ✓          | ✗           | ✗           | ✓              | ✗             |
| LAVAD [67]          | N/A            | N/A         | ✓          | ✗           | ✓           | ✗              | ✗             |
| VAD-VideoLLama [36] | 13/7           | 2,400       | ✗          | ✗           | ✓           | ✗              | projection    |
| CUVA [9]            | 11             | 6,000       | ✗          | ✗           | ✓           | ✗              | ✗             |
| Hawk [47]           | -              | 16,000      | ✗          | ✗           | ✓           | ✗              | projection    |
| HIVAU-70k (Ours)    | 19             | 70,000      | ✓          | ✓           | ✓           | ✓              | LoRA          |

method in anomaly detection within complex real-world scenarios, with the sampled frames being concentrated in anomalous regions.

## C. Discussion with related works.

In Table A, we provide a comprehensive comparison with related works in terms of benchmarks and methods.

Summary of related works: Recently, there has been substantial research on multi-modal Video Anomaly Understanding, making significant contributions to advancing open-world anomaly understanding. LAVAD [67] utilized several pre-trained foundational models to offer a trainingfree explainable VAD process. VAD-VideoLLaMA [36], designed a three-phase training method to finetune VideoLLaMA in the VAD domain. CUVA [9] introduced a dataset and metric for evaluating causation understanding of video anomalies. Hawk [47] constructed an instruction dataset and finetuned a video-language framework that incorporates both motion and video information.

## Difference and Advantages of our proposed benchmark and method:

- We develop a semi-automated annotation engine that scales hierarchical anomaly annotation efficiently, combining manual refinement with LLM-based annotation to maintain high-quality data across multiple granularities, resulting in over 70,000 annotations at clip, event, and video levels, which significantly surpasses previous

datasets in scale.

- UCA [65] only provides clip-level captions, overlooking the understanding of anomalies across longer time spans. CUVA [9] and Hawk [47], on the other hand, only offer video-level instruction data, neglecting finer-grained visual perception and anomaly analysis. In contrast, our proposed HIVAU-70k takes a multi-temporal granularity perspective, offering more comprehensive and diverse anomaly annotations for open-world anomaly detection. It enables progressive and comprehensive learning, from short-term visual perception to long-term anomaly reasoning.
- We propose the Anomaly-focused Temporal Sampler (ATS), integrated with a multi-modal visual-language model. Benefiting from the precise temporal annotations we provide, the ATS is able to focus on anomaly-dense video segments. This integration significantly improves efficiency and accuracy in long-video anomaly detection.

## D. More Qualitative Results.

As shown from Fig. G to Fig. J, we present the output of explainable text generated by Holmes-VAU compared with the base model, i.e., InternVL-2B [6]. The results demonstrate significant improvements in the model's visual perception and anomaly analysis capabilities after fine-tuning on HIVAU-70k.

## E. Limitations and Future Work.

While our work demonstrates significant strides in multigranular video anomaly understanding, several limitations present opportunities for future enhancement. First, optimizing for real-time streaming remains a challenge. Our sparse sampling approach improves efficiency, but further refinement is necessary for seamless deployment in streaming contexts. Additionally, our work has so far focused on surveillance data, extending our framework to other domains, such as industrial monitoring and medical diagnostics, will help validate its generalization capabilities. Lastly, integrating additional sensory data, like audio, and scalable hierarchical annotation could enhance anomaly detection and broaden applicability.

![Image](artifacts/image_000010_e33dd795a830a77eb5596b6dcaacbfd39aadc153e19cfc647cee2529316f011c.png)

## Ground Truth:

A green car driving on a road in rainy weather . the windshield wipers are in motion, and the road appears wet . there are traffic signs and other vehicles visible in the background .

## InternVL2

Throughout the video, we see cars driving on the road, a green car driving on the road, a green car driving on the road, a green car driving on the road, a green car driving on the road

## Holmes -VAU (Ours)

A view from inside a vehicle , looking out through the windshield at a street scene . the sky is overcast , and the visibility is low due to the weather conditions . there are several cars on the road, including a green car in the foreground . the road is lined with traffic signs and streetlights, and there are buildings and trees visible in the background .

Prompt: Describe the main events that take place in this video .

![Image](artifacts/image_000011_c8e089f186b3d9039004571460bf487905cf4e07f87e26d59defd82b2545bb35.png)

## Ground Truth:

A man in a striped shirt holding another person by the hair, appearing to be in distress . the scene transitions to a close -up of the man's hands as he picks up a knife from a cluttered kitchen counter .

## InternVL2

The video clip shows a man in a striped shirt cutting a piece of meat with a knife on a kitchen counter . The camera focuses on the man's hands and the knife as he cuts the meat . The background is blurred, emphasizing the action in the foreground .

## Holmes -VAU (Ours)

A close -up of two individuals engaged in a physical altercation . one person is wearing a striped shirt and appears to be grappling with another individual who is dressed in dark clothing . the setting is dimly lit, suggesting an indoor environment, possibly a room or a kitchen

Figure G. Qualitative comparison of anomaly understanding explanation with our baseline model, i.e., InternVL-2B. Correct and wrong explanations are highlighted in green and red, respectively.

clip-level &lt;Caption&gt;

Figure H. Qualitative comparison of anomaly understanding explanation with our baseline model, i.e., InternVL-2B. Correct and wrong explanations are highlighted in green and red, respectively.

![Image](artifacts/image_000012_81b98ed2a7732cb02e9136bb2e8331eff4b76545fd60449a56fb9ccef727ca8a.png)

Figure I. Qualitative comparison of anomaly understanding explanation with our baseline model, i.e., InternVL-2B. Correct and wrong explanations are highlighted in green and red, respectively.

![Image](artifacts/image_000013_2d1506e4bde95ee4698d65d7fca5d0b42822faf1a7655dd18a51d17756aa0199.png)

Figure J. Qualitative comparison of anomaly understanding explanation with our baseline model, i.e., InternVL-2B. Correct and wrong explanations are highlighted in green and red, respectively.

![Image](artifacts/image_000014_fb34db87c50972a28a0cb4c1c5afe48c1bd1126f21184f296f9eec454bfd4973.png)