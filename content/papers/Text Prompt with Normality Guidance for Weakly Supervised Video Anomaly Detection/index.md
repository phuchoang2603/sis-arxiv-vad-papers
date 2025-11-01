---
title: Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly 
  Detection
type: method
categories:
- Hybrid
github_link:
summary: Proposes a novel pseudo-label generation and self-training 
  framework incorporating CLIP for text-image alignment, learnable text prompts,
  normality visual prompts, a pseudo-label generation module guided by normality
  clues, and a self-adaptive temporal dependence learning module, achieving 
  state-of-the-art performance on benchmark datasets.
benchmarks:
- ucf-crime
- xd-violence
authors:
- Zhiwei Yang
- Jing Liu
- Peng Wu
date: '2023-10-01'
---

## Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection

Zhiwei Yang 1 , Jing Liu 1* *, Peng Wu 2

Guangzhou Institute of Technology, Xidian University, Guangzhou, China 2 School of Computer Science, Northwestern Polytechnical University, Xi'an, China

1

{zwyang97, neouma}@163.com, xdwupeng@gmail.com

## Abstract

Weakly supervised video anomaly detection (WSVAD) is a challenging task. Generating fine-grained pseudo-labels based on weak-label and then self-training a classifier is currently a promising solution. However, since the existing methods use only RGB visual modality and the utilization of category text information is neglected, thus limiting the generation of more accurate pseudo-labels and affecting the performance of self-training. Inspired by the manual labeling process based on the event description, in this paper, we propose a novel pseudo-label generation and selftraining framework based on Text Prompt with Normality Guidance (TPWNG) for WSVAD. Our idea is to transfer the rich language-visual knowledge of the contrastive language-image pre-training (CLIP) model for aligning the video event description text and corresponding video frames to generate pseudo-labels. Specifically, We first fine-tune the CLIP for domain adaptation by designing two ranking losses and a distributional inconsistency loss. Further, we propose a learnable text prompt mechanism with the assist of a normality visual prompt to further improve the matching accuracy of video event description text and video frames. Then, we design a pseudo-label generation module based on the normality guidance to infer reliable frame-level pseudo-labels. Finally, we introduce a temporal context self-adaptive learning module to learn the temporal dependencies of different video events more flexibly and accurately. Extensive experiments show that our method achieves state-of-the-art performance on two benchmark datasets, UCF-Crime and XD-Violence, demonstrating the effectiveness of our proposed method.

## 1. Introduction

Anomaly detection has been widely researched and applied in various fields, such as computer vision [23 , 35 , 40 , 43 ,

* Corresponding authors.

Figure 1. Illustration of the manual video frame labeling process.

![Image](artifacts/image_000000_6c82986e0f2166365b23661569d86b038ea7f626135687e341d5116dae4236bb.png)

49], natural language processing [1], and intelligent optimization [29]. One of the most important research issues is the video anomaly detection (VAD). The main purpose of VAD is to automatically identify events or behaviors in the video that are inconsistent with our expectations.

Due to the rarity of anomalous events and the difficulty of frame-level labeling, current VAD methods focus on semi-supervised [14 , 16 , 18] and weakly supervised [11 , 26 , 52] paradigms. Semi-supervised VAD methods aim to learn normality patterns from normal data, and deviations from this pattern are considered as anomalies. However, due to the lack of discriminative anomaly information in the training phase, these models are often prone to overfitting, leading to poor performance in complex scenarios. Subsequently, weakly supervised video anomaly detection (WSVAD) methods came into prominence. WSVAD involves both normal and abnormal videos with video-level labels in the training phase, but the exact location of abnormal frames is unknown. Current WSVAD methods mainly include one-stage methods based on multi-instance learning (MIL) [17 , 26 , 27] and two-stage methods based on pseudolabel self-training [6 , 11 , 51 , 53]. While the one-stage methods based on MIL show promising results, this paradigm tends to focus on video snippets with prominent anomalous features and suboptimal attention to minor anomalies, thus limiting its further performance improvement.

In contrast to the one-stage methods mentioned above, two-stage methods based on pseudo-label self-training generally use an off-the-shelf classifier or MIL to obtain initial

pseudo-labels, and then train the classifier with further refined pseudo-labels. Because these methods train the classifier directly with the generated fine-grained pseudo-labels, they show great potential in performance. However, these methods still have two aspects that have not been considered: first, the generation of pseudo-labels is based only on visual modality and lacks the utilization of textual modality, which limits the accuracy and completeness of the generated pseudo-labels. Second, the mining of temporal dependencies among video frames is insufficient.

To further exploit the potential of pseudo-label-based self-training on WSVAD, we dedicate to investigating the two problems mentioned above in this paper. Our motivation for the first question is that we explore how the textual modal information can be effectively utilized to assist in generating pseudo-labels. Recalling our manual process of video frame labeling, we mainly based on textual definitions of anomalous events, i.e., prior knowledge about anomalous events, to accurately locate the video frames. As illustrated in Fig. 1, assuming that we need to annotate the abnormal video frames that contain "fighting" event, we will first associate the textual definition of "fighting" and then look for matching video frames, which is actually a process of text-image matching based on prior knowledge. Inspired by this process, we associate a highly popular and powerful contrastive language-image pre-training (CLIP) [19] model to assist us in achieving this goal. On the one hand, the CLIP learns a large number of image-text pairs on the web, and thus has a highly rich prior knowledge; on the other hand, the CLIP is trained by comparative learning, which empowers it with excellent image-text alignment capabilities. For the second motivation, because different video events have diverse durations, this leads to different ranges of temporal dependencies. Existing methods either do not consider temporal dependencies or only consider dependencies within a fixed temporal range, leading to inadequate modeling of temporal dependencies. Therefore, in order to achieve more flexible and adequate modeling of temporal dependencies, we should investigate methods that can adaptively learn temporal dependencies of different lengths.

Based on the above two motivations, we propose a novel pseudo-label generation and self-training framework based on Text Prompt with Normality Guidance (TPWNG) for WSVAD. Our main idea is to utilize the CLIP model to match the textual descriptions of video events with the corresponding video frames, and then infer the pseudo-labels from match similarities. However, since the CLIP model is trained at the image-text level, it may suffer from domain bias and lacks the ability to learn temporal dependencies in videos. In order to better transfer the prior knowledge of CLIP to the WSVAD task, we first construct a contrastive learning framework by designing two ranking losses and a distributional inconsistency loss to fine-tune the CLIP

model for domain adaptation under the weakly-supervised setting. To further improve the accuracy of aligning the descriptive text of video events with video frames, we employ learnable textual prompts to facilitate the text encoder of CLIP to generate more generalized textual embedding features. On this basis, we propose a normality visual prompt (NVP) mechanism to aid this process. In addition, because abnormal videos contain normal video frames as well, we design a pseudo-label generation (PLG) module based on normality guidance, which can reduce the interference caused by individual normal video frames to the alignment of abnormal video frames, thus facilitating the obtaining of more accurate frame-level labels.

Furthermore, to compensate for the lack of temporal relationship modeling in CLIP as well as to more flexible and adequately mine the temporal dependencies between video frames, we introduce a temporal context self-adaptive learning (TCSAL) module for temporal dependency modeling, inspired by the work [25]. TCSAL allows the attention module in the Transformer to adaptively adjust the attention span according to the inputs by designing a temporal span adaptive learning mechanism. This can facilitate the model to capture the temporal dependencies of video events of different durations more accurately and flexibly.

Overall, our main contributions are summarized below:

- We propose a novel framework, i.e., TPWNG, to perform pseudo-label generation and self-training for WSVAD. TPWNG fine-tunes CLIP with the designed ranking loss and distributional inconsistency loss to transfer its strong text-image alignment capability to assist pseudo-label generation by means of the PLG module.
- We design a learnable text prompt and normality visual prompt mechanism to further improve the alignment accuracy of video events description text and video frames.
- We introduce a TCSAL module to learn the temporal dependencies of different video events more flexibly and accurately. To the best of our knowledge, we are the first to introduce the idea of self-adaptive learning of temporal context dependencies for VAD.
- Extensive experiments have been conducted on two benchmark datasets, UCF-Crime and XD-Violence, where the excellent performance demonstrates the effectiveness of our method.

## 2. Related Work

## 2.1. Video Anomaly Detection

The VAD task has been widely focused and researched, and many methods have been proposed to solve this problem. According to different supervision modes, these methods can be mainly categorized into semi-supervised-based and weakly supervised-based VAD.

Semi-supervised VAD. Early researchers mainly used

semi-supervised approaches to solve the VAD problem [2 , 7 , 8 , 10 , 14 , 15 , 20 , 24 , 31 , 33 , 41 – 44 , 46 , 50]. In the semi-supervised setting, only normal data can be acquired in the training phase, which aims to build a model that can characterize normal behavioral patterns by learning normal data. During the testing phase, data that contradict with the normal patterns are considered anomalies. Common semi-supervised VAD methods mainly include one-class classifier-based [21 , 33 , 37] and reconstruction [8 , 38] or prediction errors-based methods [14 , 42]. For example, Xu et al. [38] used multiple one-classifiers to predict anomaly scores based on appearance and motion features. Hasan et al. [8] built a fully convolutional auto-encoder to learn regular patterns in the video. Liu et al. in [14] proposed a novel video anomaly detection method that utilizes the U-Net architecture to predict future frames, where frames with large prediction errors are considered as anomalous.

Weakly Supervised VAD. Compared to semisupervised VAD methods, WSVAD can utilize both normal and anomalous data with video-level labels in the training phase, but the exact frame location where the abnormal event occurred is unknown. In such a setting, the one-stage approaches based on MIL [3 – 5 , 13 , 17 , 22 , 26 , 27 , 32 , 34 , 45 , 54] and the twostage approaches based on pseudo-labels self-training [6 , 11 , 51 , 53] are the two prevailing approaches. For example, Sultani et al. [26] first proposed a deep MIL ranking framework for VAD, where they considered anomalous and normal videos as positive and negative bags, respectively, and the snippets in the videos are considered as instances. Then a ranking loss is used to constrain the snippets with the highest anomaly scores in the positive and negative bags to stay away from each other. Later, many variants of the method were proposed on this basis. For example, Tian et al. [27] proposed a top-k MIL based VAD method with robust temporal feature magnitude learning.

However, these one-stage methods generally use a MIL framework, which leads to models that tend to focus only on the most significant anomalous snippets while ignoring nontrivial anomalous snippets. A two-stage approach based on pseudo-label self-training provides a relatively more promising solution. The two-stage approach first generates initial pseudo-labels using MIL or an off-the-shelf classifier and then refines the labels before using them for supervised training of the classifier. For example, Zhong et al. in [53] reformulated the WSVAD problem as a supervised learning task under noisy labels obtained by an off-the-shelf video classifier. Feng et al. in [6] introduced a multiple instance pseudo label generator that produces more reliable pseudo labels for fine-tuning a task-specific feature encoder with self-training mechanism. Zhang et al. in [51] exploited completeness and uncertainty properties to enhance pseudo labels for effective self-training. How- ever, all these existing methods only generate pseudo-labels based on visual unimodal information and lack the utilization of textual modal. Therefore, in this paper, we endeavor to combine both visual and textual modal information in order to generate more accurate and complete pseudo-labels for self-training of the classifier.

## 2.2. Large Vision-Language Models

Recently, there has been an emergence of large visionlanguage models that learn the interconnections between visual and textual modalities by pre-training on large-scale datasets. Among these methods, the CLIP demonstrates unprecedented performance in many visual-language downstream tasks, e.g. image classification [55], object detection [56], semantic segmentation [12] and so on. The CLIP model has recently been successfully extended to the video domain as well. VideoCLIP [39] is proposed to align video and textual representations by contrasting temporally overlapping video-text pairs with mined hard negatives. ActionCLIP [30] formulated the action recognition task as a multimodal learning problem rather than a traditional unimodal classification task. However, there are fewer attempts to utilize CLIP models to solve VAD tasks. Joo et al. in [9] simply utilizes CLIP's image encoder for extracting more discriminative visual features and does not use textual information. Wu et al. [36], Zanella et al. [48] mainly use textual features from CLIP to enhance the expressiveness of the overall features, followed by MIL-based anomaly classifier learning. The major difference with the above works is that our method is the first to utilize the textual features encoded by the CLIP text encoder in conjunction with the visual features to generate pseudo-labels, and then employ a supervised approach to train an anomaly classifier.

## 3. Method

In this section, we first present the definition of the WSVAD task, then introduce the overall architecture of our proposed method, and subsequently elaborate on the details of each module and the execution process.

## 3.1. Overall Architecture

Formally, we first define sets D a = {(v a i , yi)} M i=1 and D n = {(v n i , yi)} M i=1 containing M abnormal and normal videos with ground-truth labels, respectively. For each v a i
, it is labeled yi = 1, indicating that this video contains at least one anomalous video frame, but the exact location of the anomalous frame is unknown. For each v n i , it is labeled yi = 0, indicating that this video consists entirely of normal frames. With this setting, WSVAD task is to utilize coarsegrained video-level labels to enable a classifier to learn to predict fine-grained frame-level anomaly scores.

Fig. 2 illustrates the overall pipeline of our approach. Normal and abnormal video along with learnable category

Figure 2. The overall architecture of our proposed TPWNG.

![Image](artifacts/image_000001_828a7324e2e42726199da3d2556e380003090219b876f71e2bf9d5aa4f8b809e.png)

prompt text are encoded as feature embedding by the image encoder and text encoder of CLIP, respectively. Then, the text encoder of CLIP is encouraged by fine-tuning it to produce textual feature embedding of video event categories that accurately match anomalous or normal video frames, and the NVP assists in this process. Meanwhile, the image features feed the TCSAL module to perform self-adaptive learning of temporal dependencies. Finally, a video frame classifier is trained to predict anomaly scores under the supervision of pseudo-labels obtained by the PLG module.

## 3.2. Text and Normality Visual Prompt

Learnable Text Prompt. Constructing textual prompts that can accurately describe various video event categories is a prerequisite for realizing the alignment of text and corresponding video frames. However, it is impractical to manually define description texts that can completely characterize anomalous events in all different scenarios. Therefore, inspired by CoOp [55], we employ a learnable text prompt mechanism to adaptively learn representative video event text prompts to align the corresponding video frames. Specifically, we construct a learnable prompt template, which adds l learnable prompt vectors in front of the tokenized category name, as follows:

<!-- formula-not-decoded -->

where ∂ l denotes the l − th prompt vector. Tokenizer is converting original category labels, i.e., "fighting", "accident", . . . , "normal", etc., into class tokens by means of

CLIP tokenizer. Then, we add the corresponding location information pos to the learnable prompts and then feed it to the CLIP text encoder ζtext to get the feature embedding Tl Tlabel ∈ R D of the video event description text as follows:

<!-- formula-not-decoded -->

Finally, we compute all video event categories according to Eqs. (1) and (2) to obtain the video event description text embedding set E = {T
1 a T
1 , T 
2 a T 
2 , ..., T 
k a T 
k − 1 , T 
k n T 
k }, where {T
i a T
i } k − 1 i=1 denotes the description text embedding of preceding k − 1 abnormal events and T
k n T
k denotes the description text embedding of normal events.

Normality Visual Prompt. For an anomalous video, which contains both anomalous and normal frames, our core task is to infer pseudo-labels from the match similarities between the description text of the anomalous events and the video frames. However, this process is susceptible to interference from normal frames in the anomalous video because they have a similar background to the anomalous frames. To minimize this interference, we propose a NVP mechanism. NVP is used to assist the normal event description text to more accurately align normal frames in the abnormal video, and thus indirectly assist the description text of abnormal event to align abnormal video frames in the abnormal video by means of the distribution inconsistency loss that will be introduced in Sec. 3.5. Specifically, we first compute the match similarities S nn i, k ∈ R F between the description text embedding of normal event and the video frame features in the normal video. Then, the match similarities after softmax operation are used as weights to aggre-

gate normal video frame features to obtain NVP Qi ∈ R D . The formulas are represented as follows:

<!-- formula-not-decoded -->

where X n i ∈ R F ×D denotes the visual features of the normal video v n i obtained by the CLIP image encoder, where F and D denote the number of video frames and feature dimensions, respectively. Then, we concatenate Qi and T
k n T
k in the feature dimension and feed an FFN layer with skip connections to obtain the enhanced description text embedding T
k ˙ n T
k of normal events. The formula is represented as follows:

<!-- formula-not-decoded -->

## 3.3. Pseudo Label Generation Module

In this subsection, we detail how to generate frame-level pseudo labels. For a normal video, we can directly get the frame-level pseudo-labels, i.e., for a v n i = {Ij} F j=1 containing F normal frames, it corresponds to a label set {γ n i, j = 0} F j=1 . Our main goal is to infer the pseudo-labels for anomalous videos that contain both anomalous and normal frames. To this end, we propose a PLG module for inferring accurate pseudo-labels based on the normality guidance. PLG module infers frame-level pseudo-labels by incorporating the match similarities between the description text of the normal event and the abnormal video as a guide into the match similarities between the description text of the corresponding abnormal event and the abnormal video.

Specifically, we first compute the match similarities S an i, k = X a i (T
k ˙ n T
k ) ⊤ between normal event description text embedding enhanced with NVP and anomalous video features, where X a i ∈ R F ×D denotes the visual features of the anomalous video v a i obtained by the CLIP image encoder. Similarly, we compute the match similarities S aa i, τ = X a i (T
τ a T
τ ) ⊤ between the description text embedding T
τ a T
τ of the corresponding τ -th (1 ⩽ τ ⩽ k − 1) real anomaly category and the anomaly video features X a i .

Theoretically, for S aa i, τ , it should have high match similarities corresponding to abnormal frames and low match similarities for normal frames. But it may be interfered by normal frames from the same video having the same background. To reduce the interference of normal frames, we infer pseudo-labels by incorporating the matching similarity corresponding to the description text of normal events with certain weights as a guide into the matching similarity of the description text of corresponding real abnormal events. Specifically, we first perform a normalization and fusion operation on S aa i, τ and S an i, k as follows:

<!-- formula-not-decoded -->

where ˜ ∗ denotes the normalization operation and α denotes the guidance weight. After obtaining ψi, we similarly perform a normalization operation on it to obtain ψ ˜ i . Then, we set a threshold θ on ψ ˜ i to obtain the frame-level pseudolabels in the anomalous video as follows:

<!-- formula-not-decoded -->

where γ a i, j denotes the pseudo-label of the j-th frame in the i-th anomaly video. Finally, we combine the framelevel pseudo-labels γ n i, j and γ a i, j of normal and anomalous videos to get the total pseudo-label set {γi, j} F j=1 .

## 3.4. Temporal Context Self-adaptive Learning

To adaptively adjust the learning range of temporal relationship based on the input video data, inspired by the work [25], we introduce a TCSAL module. The backbone of TCSAL is the transformer-encoder, but unlike the original transformer, the spanning range of attention is controlled by a soft mask function χ z for each self-attention head at each layer. χ z is a piecewise function mapping a distance to a value between [0, 1] as follows:

<!-- formula-not-decoded -->

where h represents the distance between the current t-th frame in a video and the r − th (r ∈ [1, t − 1]) frame in the past temporal range. R is a hyperparameter used to control the softness. z is a learnable parameter that is adaptively tuned with the input as follows:

<!-- formula-not-decoded -->

here σ represents the sigmoid operation, C and b are learnable parameters during model training. With the soft mask function χ z , the corresponding attention weights ωt, r is computed within this mask, i.e.,

<!-- formula-not-decoded -->

here βt,r denotes the dot product output of the Query corresponding to the t-th frame in a video with the Key corresponding to the r−th frame in the past. Under the control of χz, the self-attention heads will be able to adaptively adjust the self-attention span range according to the input.

Finally, the video features after temporal context adaptive learning are fed into a classifier to predict the framelevel abnormality scores {ηi, j} F j=1 .

## 3.5. Objective Function

First, we fine-tune the CLIP text encoder. For a normal video, we further compute the match similarities set φ na i = {S na i, τ = X n i (T
τ a T
τ ) ⊤ |1 ⩽ τ ⩽ k − 1} between the description texts of the other k − 1 anomalous events and the normal frames. We expect that the maximum in the similarity set

φ na i should be as small as possible while the maximum in S nn i, k should be as large as possible. Thus, we design the following ranking loss for constraints:

<!-- formula-not-decoded -->

For an anomalous video, we first calculate the similarities S an i, k = X a i (T
k ˙ n T
k ) ⊤ between the description text embedding of normal event and the anomalous video features, the similarity S aa i, τ = X a i (T
τ a T
τ ) ⊤ between the description text embedding of the τ -th (1 ⩽ τ ⩽ k − 1) real anomalous event category and the anomalous video features, and the similarity set φ aa i = {S aa i, g = X a i (T
g a T
g ) ⊤ |1 ⩽ g ⩽ k − 1, g ̸= τ } between the description text embedding of other k − 2 anomalous event categories and the anomalous video features, respectively. We expect that the maximum value in S an i, k should be greater than the maximum value in φ aa i . Similarly, the maximum value in S aa i, τ should be greater than the maximum value in φ aa i . In short, it means that we expect that the description texts of real abnormal and normal events should match the abnormal and normal frames in the abnormal video with the highest possible similarity, respectively. Thus, the ranking loss for anomalous videos is designed as follows:

<!-- formula-not-decoded -->

In addition, to further ensure that the description texts of real abnormal events and normal events can accurately align the abnormal and normal video frames in the abnormal video, respectively, we design a distribution inconsistency loss (DIL). DIL is used to constrain the similarities between the description text of the real abnormal event and the video frames to be inconsistent with the similarity distribution between the description text of the normal event and the video frames. We use cosine similarity to perform this loss:

<!-- formula-not-decoded -->

Then, following the work [26], in order to make the generated pseudo-labels satisfy sparsity and smoothing in temporal order, we impose sparsity and smoothing constraints, L sp = P F j=1 (S ˜ aa i, j, τ − S ˜ aa i, j+1, τ ) 2 , L sm = P F j=1 S ˜ aa i, j, τ , on the similarity vectors S ˜ aa i, τ .

Then, we calculate the binary cross-entropy between the anomaly score ηi,j predicted by the classifier and the pseudo-label γi, j as the classification loss:

<!-- formula-not-decoded -->

The final overall objective function balanced by λ1 and λ 2 is designed as follows:

<!-- formula-not-decoded -->

## 4. Experiments

## 4.1. Datasets and Evaluation Metrics

Datasets. We conduct extensive experiments on two benchmark datasets, UCF-Crime [26] and XD-Violence [34]. UCF-Crime is a large-scale real scene dataset for WSVAD. UCF-Crime duration is 128 hours in total and contains 1900 surveillance videos covering 13 anomaly event categories, of which 1610 videos with video-level labels are used for training and 290 videos with frame-level labels are used for testing. XD-Violence is a large-scale violence detection dataset collected from movies, online videos, surveillance videos, CCTVS, etc. XD-Violence lasts 217 hours and contains 4754 videos covering 6 anomaly event categories, of which 3954 training videos with video-level labels and 800 test videos with frame-level labels.

Evaluation Metrics. Following the previous methods [6 , 26], for the UCF-Crime dataset, we measure the performance of our method using the area under the curve (AUC) of the frame-level receiver operating characteristics (ROC). Similarly, for the XD-Violence dataset, we follow the evaluation criterion of average precision (AP) suggested by the work [34] to measure the effectiveness of our method.

## 4.2. Implementation Details

The image and text encoders in our method use a pre-trained CLIP (VIT-B/16), in which both the image and text encoders are kept frozen, except for the text encoder where the final projection layer is unfrozen for fine-tuning. The feature dimension D is 512. FFN is a standard block from Transformer. The length l of the learnable sequence in the text prompt is set to 8. The normality guidance weight α is set to 0.2 for both the UCF-Crime and XDViolence datasets. The pseudo-labels generation threshold θ is set to 0.55 and 0.35 for the UCF-Crime and XD-Violence datasets, respectively. The parameter R used to control the softness of the soft mask function is set to 256. The sparse loss and smoothing loss weights are set to λ1 = 0 . 1 and λ 2 = 0 . 01. Please refer to the supplementary materials for more details on implementation.

## 4.3. Comparison with State-of-the-art Methods

We compare the performance on the UCF-Crime and XDViolence datasets with the current state-of-the-art (SOTA) methods in Tab. 1. As can be observed from the table, our method achieves a new SOTA on both the UCF-Crime and XD-Violence datasets. Specifically, for the UCF-Crime dataset, our method outperforms the current SOTA method

Table 1. AUC and AP on UCF-Crime and XD-Violence dataset.

|        | Methods            | UCF (AUC)    | XD (AP)   |
|--------|--------------------|--------------|-----------|
| Weakly | Sultani et al.[26] | 77.92%       | 73.20%    |
| Weakly | GCN [53]           | 82.12%       | -         |
| Weakly | HL-Net [34]        | 82.44%       | 73.67%    |
| Weakly | CLAWS [45]         | 82.30%       | -         |
| Weakly | MIST [6]           | 82.30%       | -         |
| Weakly | RTFM [27]          | 84.30%       | 77.81%    |
| Weakly | CRFD [32]          | 84.89%       | 75.90%    |
| Weakly | GCL [47]           | 79.84%       | -         |
| Weakly | MSL [11]           | 85.62%       | 78.58%    |
| Weakly | MGFN [3]           | 86.67%       | 80.11%    |
| Weakly | Zhang et al.[51]   | 86.22%       | 78.74%    |
| Weakly | UR-DMU [54]        | 86.97%       | 81.66%    |
| Weakly | CLIP-TSA [9]       | 87.58%       | 82.17%    |
| Weakly | Ours               | 87.79%       | 83.68%    |

CLIP-TSA [9] by 0.21%, which is not a trivial improvement for the challenging WSVAD task. Most importantly, compared to methods MIST [6] and Zhang et al. [51] similar to ours that also use pseudo-label-based self-training, our method significantly outperforms them by 5.49% and 1.57%, respectively. This fully demonstrates that our proposed pseudo-label generation and self-training framework is vastly superior to the above two approaches. This also indicates that transferring visual language multimodal associations through CLIP is conducive to generating more accurate pseudo-labels compared to merely utilizing unimodal visual information. For the XD-Violence dataset, our method also surpasses the current optimal method CLIPTSA [9] by 1.52%. Compared to a similar pseudo-labelbased self-training method Zhang et al. [51], our method also outperforms it by 4.94%. The consistent superior performance on two large-scale real datasets strongly demonstrates the effectiveness of our method. This also shows the extraordinary potential of the pseudo-label based selftraining scheme, if accurate pseudo-labels can be generated utilizing multiple modality information.

## 4.4. Ablation Studies

We conduct ablation experiments in this subsection to analyze the effectiveness of each component of our framework.

Effectiveness of Normal Visual Prompt. To verify the validity of NVP, we execute three comparison experiments: without NVP, with NVP based on frame averaging (NVPFA), and with NVP based on match similarities aggregation (NVP-AS). As can be seen from the results in Tab. 2, in the absence of NVP, the performance of our method on the UCF-Crime and XD-Violence datasets decreases by 2.54% and 2.10% compared to with an NVP-AS, respectively. NVP-AS boosts the performance of the method by 0.47% and 0.55% more compared to NVP-FA on UCF-Crime and

Table 2. The AUC and AP of our method on the UCF and XD datasets without NVP, with NVP-FA, and with NVP-AS.

| UCF-Crime (AU   | UCF-Crime (AUC)    | XD-Violence (AP)   |
|-----------------|--------------------|--------------------|
| 85.25%          | 81.58%             | w/o NVP            |
| 87.32%          | 83.13%             | w NVP-FA           |
| 87.79%          | 83.68%             | w NVP-AS           |

Table 3. The AUC and AP of our method on the UCF and XD datasets with NG and without NG.

|        | UCF-Crime (AUC   | XD-Violence (AP)   |
|--------|------------------|--------------------|
| 85.83% | 81.32%           | w/o NG             |
|        | 83.68%           | w NG               |

XD-Violence datasets, respectively. This reveals two facts: first, NVP can help the text embedding to better match normal frames in anomalous videos, which indirectly aids in generating more accurate pseudo-labels in cooperation with the DIL and the normality guidance mechanism. Second, the NVP-AS can effectively reduce the interference of some noise snippets (e.g., prologue, perspective switching, etc.) in normal videos compared to the NVP-FA approach, thus obtaining a purer NVP.

Effectiveness of the Normality Guidance. In the pseudo-label generation module, instead of inferring pseudo-labels directly based on the similarity between the corresponding abnormal event description text and the abnormal video, we incorporate guidance from the match similarities of the normal event description text counterparts, aiming to reduce the interference of partially noisy video frames and generate more accurate pseudo-labels. To verify the contribution of the normality guidance, we compare the impact of the pseudo-label generation module on the performance of our method with and without normal guidance (NG), respectively. As can be observed from Tab. 3, when our method is equipped with normal guidance, the performance rises by 1.96% and 2.36% on the UCF-Crime and XD-Violence datasets, respectively. This validates the effectiveness of the normality guidance.

Effectiveness of TCSAL. To analyze the effectiveness of TCSAL module, we conduct comparative experiments with the Transformer-encoder (TF-encoder) module in [28], MTN module in [27], and GL-MHSA module in [54] by replacing the temporal learning module in our framework with each of these three modules. From Tab. 4, it can be observed that the TF-encoder module has the lowest performance, which is understandable since the global selfattention computation way makes it neglect to pay attention to the local temporal information. Both MTN and GLMHSA outperform TF-encoder with comparable performance. Our introduced TCSAL module achieved the best performance on both datasets. This indicates that adopting

![Image](artifacts/image_000002_fba442e0c2ff15d0401302256d05767996469ec7515cbc4615d45e880d9d9a1c.png)

Figure 3. Anomaly score curves of several test samples on the UCF-Crime and XD-Violence dataset.

Table 4. The AUC and AP of our method on the UCF and XD datasets with different temporal modules.

|              | UCF (AUC)    | XD (AP)   |
|--------------|--------------|-----------|
| w TF-encoder | 85.12%       | 80.02%    |
| w MTN        | 86.22%       | 81.02%    |
| w GL-MHSA    | 86.43%       | 81.23%    |
| w TCSAL      | 87.79%       | 83.68%    |

Table 5. Comparison of the AUC and AP of our method with different loss terms on the UCF-Crime and XD-Violence datasets. "bs" indicates that Lcl , L sp, L sm three loss functions are used.

| Loss term    | Loss term    | Loss term    | Loss term    | Dataset   | Dataset   |
|--------------|--------------|--------------|--------------|-----------|-----------|
| bs           | Ln
 rank     | L
 a
 rank   | Ldil         | UCF (AUC) | XD (AP)   |
| ✓            |              |              |              | 77.12%    | 73.32%    |
| ✓            | ✓            |              |              | 81.34%    | 78.67%    |
| ✓            |              | ✓            |              | 84.45%    | 81.56%    |
| ✓            |              |              | ✓            | 82.47%    | 79.96%    |
| ✓            | ✓            | ✓            | ✓            | 87.79%    | 83.68%    |

the mechanism of self-attention span range adaptive learning enables the temporal learning module to self-adapt to the inputs of videos with different event lengths, achieving more accurate modeling of temporal dependencies while weakening the interference of other non-relevant temporal information in the non-event span range.

## 4.5. Qualitative Results

We show the anomalous scores of our method on several test videos in Fig. 3. It can be obviously noticed that there is a steep rise in the anomaly scores when various anomalous events occur, and as the anomalous events end, the anomaly scores fall back to the lower range rapidly. For normal events, our method gives a lower abnormal score. This intuitively demonstrates that our method has good sensitivity to abnormal events and can accurately and timely detect the occurrence of abnormal events while maintaining a low abnormal score prediction for normal events.

## 4.6. Analysis of Losses

To analyze the impact of the three loss functions L n rank, L a rank , and Ldil, we perform ablation experiments on the UCF-Crime and XD-Violence datasets. As shown in Tab. 5 , when all three loss functions are absent, the performance of our method is unsatisfactory. This reveals that the original CLIP suffers from domain bias and is not directly applicable to the VAD domain. When three loss functions are available individually, the performance of our method is clearly improved, where the L a rank gives the biggest boost to the performance. When all three losses are combined and cooperate with each other, our method achieves the best performance. This demonstrates the effectiveness of the three loss functions we have designed, and they can effectively assist CLIP in domain adaptation for WSVAD.

## 5. Conclusions

In this paper, we propose a novel framework, TPWNG, to perform pseudo-label generation and self-training for WSVAD. TPWNG finetunes CLIP with the designed ranking loss and distributional inconsistency loss to transfer its textimage alignment capability to assist pseudo-label generation with the PLG module. Further, we design a learnable text prompt and normality visual prompt mechanisms to further improve the alignment accuracy of video events description text and video frames. Finally, we introduce a TCSAL module to learn the temporal dependencies of different video events more flexibly and accurately. We perform extensive experiments on the UCF-Crime and XD-Violence datasets, and the superior performance compared to existing methods demonstrates the effectiveness of our method.

## 6. Acknowledgments

This work was supported by the Guangzhou Key Research and Development Program (No. 202206030003), the Fundamental Research Funds for the Central Universities, the Innovation Fund of Xidian University (No. YJSJ24006), and the Guangdong High-level Innovation Research Institution Project (No. 2021B0909050008).

## References

- [1] Christophe Bertero, Matthieu Roy, Carla Sauvanaud, and Gilles Tredan. Experience report: Log mining using natural ´ ´ language processing and application to anomaly detection. In ISSRE, pages 351–360, 2017. 1
- [2] Ruichu Cai, Hao Zhang, Wen Liu, Shenghua Gao, and Zhifeng Hao. Appearance-motion memory consistency network for video anomaly detection. In AAAI, pages 938–946, 2021. 3
- [3] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In AAAI, pages 387–395, 2023. 3 , 7
- [4] MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, and Sangyoun Lee. Look around for anomalies: Weakly-supervised anomaly detection via context-motion relational learning. In CVPR, pages 12137– 12146, 2023.
- [5] MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, and Sangyoun Lee. Look around for anomalies: Weakly-supervised anomaly detection via context-motion relational learning. In CVPR, pages 12137– 12146, 2023. 3
- [6] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In CVPR, pages 14009–14018, 2021. 1 , 3 , 6 , 7
- [7] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In CVPR, pages 1705–1714, 2019. 3
- [8] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In CVPR, pages 733–742, 2016. 3
- [9] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In ICIP, pages 3230–3234, 2023. 3 , 7
- [10] Sangmin Lee, Hak Gu Kim, and Yong Man Ro. Bman: Bidirectional multi-scale aggregation networks for abnormal event detection. IEEE TIP, 29:2395–2408, 2019. 3
- [11] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In AAAI, pages 1395–1403, 2022. 1 , 3 , 7
- [12] Yuqi Lin, Minghao Chen, Wenxiao Wang, Boxi Wu, Ke Li, Binbin Lin, Haifeng Liu, and Xiaofei He. Clip is also an efficient segmenter: A text-driven approach for weakly supervised semantic segmentation. In CVPR, pages 15305–15314, 2023. 3
- [13] Tianshan Liu, Kin-Man Lam, and Jun Kong. Distilling privileged knowledge for anomalous event detection from weakly labeled videos. IEEE TNNLS, pages 1–15, 2023. 3
- [14] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Fu-

ture frame prediction for anomaly detection–a new baseline. In CVPR, pages 6536–6545, 2018. 1 , 3

- [15] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In ICCV, pages 341–349, 2017. 3
- [16] Hui Lv, Chen Chen, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang. Learning normal dynamics in videos with meta prototype network. In CVPR, pages 15425–15434, 2021. 1
- [17] Hui Lv, Chuanwei Zhou, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang. Localizing anomalies from weakly-labeled videos. IEEE TIP, 30:4505–4515, 2021. 1 , 3
- [18] Hyunjong Park, Jongyoun Noh, and Bumsub Ham. Learning memory-guided normality for anomaly detection. In CVPR , pages 14372–14381, 2020. 1
- [19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 2
- [20] Mohammad Sabokrou, Mahmood Fathy, Mojtaba Hoseini, and Reinhard Klette. Real-time anomaly detection and localization in crowded scenes. In CVPR, pages 56–62, 2015. 3
- [21] Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, and Ehsan Adeli. Adversarially learned one-class classifier for novelty detection. In CVPR, pages 3379–3388, 2018. 3
- [22] Hitesh Sapkota and Qi Yu. Bayesian nonparametric submodular video partition for robust anomaly detection. In CVPR , pages 3212–3221, 2022. 3
- [23] Fangtao Shao, Jing Liu, Peng Wu, Zhiwei Yang, and Zhaoyang Wu. Exploiting foreground and background separation for prohibited item detection in overlapping x-ray images. PR, 122:108261, 2022. 1
- [24] Giulia Slavic, Abrham Shiferaw Alemaw, Lucio Marcenaro, David Martin Gomez, and Carlo Regazzoni. A kalman variational autoencoder model assisted by odometric clustering for video frame prediction and anomaly detection. IEEE TIP , 32:415–429, 2022. 3

´

- [25] Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. In ACL, pages 331–335, 2019. 2 , 5
- [26] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In CVPR, pages 6479–6488, 2018. 1 , 3 , 6 , 7
- [27] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In ICCV, pages 4975–4986, 2021. 1 , 3 , 7
- [28] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017. 7
- [29] Chao Wang, Jing Liu, Kai Wu, and Zhaoyang Wu. Solving multitask optimization problems with adaptive knowledge transfer via anomaly detection. IEEE TEC, 26(2):304–318, 2021. 1

- [30] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472, 2021. 3
- [31] Xuanzhao Wang, Zhengping Che, Bo Jiang, Ning Xiao, Ke Yang, Jian Tang, Jieping Ye, Jingyu Wang, and Qi Qi. Robust unsupervised video anomaly detection by multipath frame prediction. IEEE TNNLS, 33(6):2301–2312, 2021. 3
- [32] Peng Wu and Jing Liu. Learning causal temporal relation and feature discrimination for anomaly detection. IEEE TIP , 30:3513–3527, 2021. 3 , 7
- [33] Peng Wu, Jing Liu, and Fang Shen. A deep one-class neural network for anomalous event detection in complex scenes. IEEE TNNLS, 31(7):2609–2622, 2019. 3
- [34] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In ECCV, pages 322–339, 2020. 3 , 6 , 7
- [35] Peng Wu, Jing Liu, Xiangteng He, Yuxin Peng, Peng Wang, and Yanning Zhang. Towards video anomaly retrieval from video anomaly detection: New benchmarks and model. arXiv preprint arXiv:2307.12545, 2023. 1
- [36] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In AAAI, pages 6074–6082, 2024. 3
- [37] Dan Xu, Elisa Ricci, Yan Yan, Jingkuan Song, and Nicu Sebe. Learning deep representations of appearance and motion for anomalous event detection. arXiv preprint arXiv:1510.01553, 2015. 3
- [38] Dan Xu, Yan Yan, Elisa Ricci, and Nicu Sebe. Detecting anomalous events in videos by learning deep representations of appearance and motion. CVIU, 156:117–127, 2017. 3
- [39] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021. 3
- [40] Minghui Yang, Jing Liu, Zhiwei Yang, and Zhaoyang Wu. Slsg: Industrial image anomaly detection by learning better feature embeddings and one-class classification. arXiv preprint arXiv:2305.00398, 2023. 1
- [41] Zhiwei Yang, Jing Liu, and Peng Wu. Bidirectional retrospective generation adversarial network for anomaly detection in videos. IEEE Access, 9:107842–107857, 2021. 3
- [42] Zhiwei Yang, Peng Wu, Jing Liu, and Xiaotao Liu. Dynamic local aggregation network with adaptive clusterer for anomaly detection. In ECCV, pages 404–421, 2022. 3
- [43] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In CVPR, pages 14592–14601, 2023. 1
- [44] Muchao Ye, Xiaojiang Peng, Weihao Gan, Wei Wu, and Yu Qiao. Anopcn: Video anomaly detection via deep predictive coding network. In ACM MM, pages 1805–1813, 2019. 3
- [45] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Claws: Clustering assisted weakly

supervised learning with normalcy suppression for anomalous event detection. In ECCV, pages 358–376, 2020. 3 , 7

- [46] Muhammad Zaigham Zaheer, Jin-Ha Lee, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Stabilizing adversarially learned one-class novelty detection using pseudo anomalies. IEEE TIP, 31:5963–5975, 2022. 3
- [47] Muhammad Zaigham Zaheer, Arif Mahmood, Muhammad Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In CVPR, pages 14744–14754, 2022. 7
- [48] Luca Zanella, Benedetta Liberatori, Willi Menapace, Fabio Poiesi, Yiming Wang, and Elisa Ricci. Delving into clip latent space for video anomaly recognition. arXiv preprint arXiv:2310.02835, 2023. 3
- [49] Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. Draem- ˇ ˇ a discriminatively trained reconstruction embedding for surface anomaly detection. In ICCV, pages 8330–8339, 2021. 1
- [50] Shuangfei Zhai, Yu Cheng, Weining Lu, and Zhongfei Zhang. Deep structured energy based models for anomaly detection. In ICML, pages 1100–1109, 2016. 3
- [51] Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang. Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection. In CVPR, pages 16271– 16280, 2023. 1 , 3 , 7
- [52] Jiangong Zhang, Laiyun Qing, and Jun Miao. Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection. In ICIP, pages 4030– 4034, 2019. 1
- [53] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In CVPR, pages 1237–1246, 2019. 1 , 3 , 7
- [54] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In AAAI, pages 3769–3777, 2023. 3 , 7
- [55] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. IJCV, V, 130(9):2337–2348, 2022. 3 , 4
- [56] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krahenb ¨ ¨ uhl, and Ishan Misra. Detecting twenty-thousand ¨ ¨ classes using image-level supervision. In ECCV, pages 350– 368, 2022. 3

## Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection

## Supplementary Material

## 7. Network Structure Details

TCSAL. The TCSAL module consists of 4 transformerencoder layers with 4 attention heads per layer, and each self-attention head of each layer is self-adaptively adjusting its attention span by a soft-masking function χ z (h). The shape of the soft mask function χ z (h) is shown in Fig. 4 . Classifier. The classifier adopts a simple structure that consists of a layer normalization layer, a linear layer, and a sigmoid layer.

## 8. Finetuning and Prompt Learning

The finetuning of the CLIP text encoder is performed together with the training of the NVP, PLG, TCSAL, and Classifier modules. During this process, the weights of both the CLIP image and text encoder are frozen, except for the last projection layer of the text encoder which is unfrozen for finetuning.

To set the optimal fine-tuning configuration, we perform finetuning experiments on the final projection layers of the CLIP image encoder and text encoder as shown in Tab. 6 . We find that based on prompt learning, there is a large performance improvement after fine-tuning the CLIP text encoder alone, whereas if we finetune only the final projection layer of the CLIP image encoder or both the image encoder and the text encoder at the same time, the performance instead decreases in both cases. Thus our final choice is prompt learning + finetuning (text encoder). We think this is due to the relatively small video anomaly dataset causing overfitting of the CLIP image encoder, which affects the method performance. When finetuning the text encoder alone, the overfitting situation is mitigated because of the prompt learning assistance. Finetuning also facilitates domain adaptation, so this combination of prompt learning + finetuning (text encoder) performs optimally.

Table 6. The AUC and AP change of our method on the UCFCrime and XD-Violence datasets with different finetuning and prompt learning configurations.

| CLIP finetuning and prompt learning configurations    | UCF (AUC)    | XD (AP)   |
|-------------------------------------------------------|--------------|-----------|
| No finetune + prompt learning                         | 86.45%       | 81.33%    |
| Image encoder finetuning + prompt learning            | 84.23%       | 81.16%    |
| Text&Image encoder finetuning + prompt learning       | 85.76%       | 82.12%    |
| Text encoder finetuning + prompt learning             | 87.79%       | 83.68%    |

Figure 4. The shape of the soft mask function χ z (h) .

![Image](artifacts/image_000003_912aa27032b62ba4336aba071b5dd6bcf4b6a450bd64750038649456c36243f4.png)

## 9. Training and Inference

Different from existing two-stage methods that separate pseudo-label generation and classifier self-training into two stages, in our approach, we synchronize pseudo-label generation and classifier training until both converge. This ensures that the updated pseudo-labels are used for supervised classifier training in real time, minimizing the interference of noisy labels on classifier training. After training under the supervision of the generated pseudo-labels, only the CLIP image encoder, the TCSAL, and the classifier are involved in the testing phase, where the video frame anomaly scores are predicted directly by the classifier.

## 10. Implementation Details.

Our method is implemented on a single NVIDIA RTX 3090 GPU using the Pytorch framework. We use Adam optimizer with a weight decay of 0.005. The batch size is set to 64, which contains 32 normal videos and 32 abnormal videos randomly sample from the training dataset. For the UCFCrime dataset, the learning rate and total epoch are set to 0.001 and 50, respectively. For the XD-Violence dataset, the learning rate and the total epoch are set to 0.0001 and 20, respectively.

## 11. Impact of Normality Guidance Weight α .

The normality guidance weight α is used to control the degree of fusion of S ˜ an i, k and S ˜ aa i, τ during pseudo-labels generation. In order to analyze the effect of α, we set different values of α for comparison experiments. As shown in Fig. 5, our method achieves optimal performance on both UCF-Crime and XD-Violence datasets when α is set to 0.2. It can be observed that as α gradually increases, the performance of our method gradually decreases, we consider that it is because too large α instead affects the alignment of the real anomaly event description text and the anomaly frames, and α = 0 . 2 is the best trade-off.

Figure 5. The AUC and AP change of our method on the UCFCrime and XD-Violence datasets with different normality guidance weight α .

![Image](artifacts/image_000004_74524aaf56b7dc5c19c08cb164e2bb4ca687f8933b44c26c908cb36543487e66.png)

Figure 6. The AUC and AP change of our method on the UCFCrime and XD-Violence datasets with different pseudo-label generation threshold θ .

![Image](artifacts/image_000005_8a3a71506d80eaa8bedeede0e0df7f8861f2110097d8853541dc93fd9b319595.png)

## 12. Impact of Pseudo-label Generation Threshold θ .

To analyze the impact of different pseudo-label generation thresholds θ on the performance of our method, we set up a series of different thresholds θ to perform comparative experiments. As shown in Fig. 6, the two datasets have different sensitivities to the threshold θ. When θ is set to 0.55 and 0.35, our method achieves the optimal performance on UCF-Crime and XD-Violence datasets, respectively.

## 13. Impact of Context Length l in Learnable Prompt.

To investigate the optimal length of learnable textual prompts, we conduct comparative experiments with the context length l being set to 4, 8, 16, and 32, respectively. As shown in Tab. 7, both datasets achieve the best performance with the context length l set to 8, and slightly lower performance with a length of 16. However, when the context length l is set to 4 or 32, the performance of our method

Table 7. The AUC and AP of our method on the UCF-Crime and XD-Violence datasets with different context lengths l .

|   l  | UCF-Crime (AUC)    | XD-Violence (AP)   |
|------|--------------------|--------------------|
|   4  | 82.26%             | 77.45%             |
|   8  | 87.79%             | 83.68%             |
|  16  | 87.24%             | 82.99%             |
|  32  | 85.23%             | 81.78%             |

Figure 7. Visualization of pseudo-labels of some video clips on the UCF-Crime dataset.

![Image](artifacts/image_000006_d01675f256ceda2f97d795d3f6c771540a238eedd9a07ccd5ee8caf9872275b0.png)

suffers a large degradation. We conjecture that the reason for this result is that too short a context length leads to textual prompts that do not fully characterize the video frame events, leading to model underfitting. Conversely, too long context length may lead to model overfitting.

## 14. Visualization of Pseudo-labels.

We visualize part of the pseudo-labels (UCF-Crime) in Fig. 7. The generated pseudo-labels (orange solid line) approximate the ground-truth (shades of blue) well in most cases, which indicates the effectiveness of the generated pseudo-labels.

## 15. Visualization of Match Similarities.

To more intuitively show that our constructed framework can facilitate the CLIP model to perform domain adaptation for matching video event text descriptions and corresponding video frames, we visualize the S
τ ˜ aa S
τ and S ˜ an k , i.e., the match similarities of real abnormal event description text and normal event description text with corresponding abnormal videos, on the UCF-Crime and XD-Violence datasets, respectively. We can observe from Fig. 8 that the distributions of S
τ ˜ aa S
τ and S ˜ an k are contradictory which can align anomalous video frames and normal video frames, respectively. This shows the effectiveness of our designed distributional inconsistency loss Ldil. In addition, we can notice from Fig. 8 (a) and (f) that there are fluctuations in the alignment of the real abnormal event description text and the corresponding abnormal video frames in these two samples, while the normal event description text has a more accurate alignment, in which case our proposed normal guidance mechanism can assist S
τ ˜ aa S
τ to better align the abnormal video frames.

Figure 8. Visualization of match similarities between video event description text and video frames for several anomaly samples on the UCF-Crime and XD-Violence test datasets. The light blue range represents abnormal ground truth.

![Image](artifacts/image_000007_cf6f8d50023f1cfa27e0c973b49c64e39a294ae6a5d0ab0cde21a28563bd0d08.png)