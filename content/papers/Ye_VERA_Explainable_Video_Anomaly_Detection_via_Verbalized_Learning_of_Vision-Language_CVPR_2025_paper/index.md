---
title: 'VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language
  Models'
type: method
categories:
- Hybrid
github_link: https://vera-framework.github.io
description: Introduces VERA, a framework that enables frozen vision-language 
  models to perform explainable video anomaly detection by learning detailed 
  anomaly-characterization questions from coarsely labeled data, without model 
  parameter modifications. The method decomposes complex reasoning into 
  reflections on guiding questions, optimizes them via verbal interactions, and 
  guides VLMs to generate segment- and frame-level anomaly scores with improved 
  explainability and performance.
benchmarks:
- ucf-crime
- xd-violence
authors:
- Muchao Ye
- Weiyang Liu
- Pan He
date: '2023-10-01'
---

![Image](artifacts/image_000000_6d1e247b7d0a98b62a35558742715e9c9ca643c3fbfd736fe597c595aff4e11b.png)

This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version;

the final published version of the proceedings is available on IEEE Xplore.

## VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

Muchao Ye 1*

Weiyang Liu 2

Pan He 3

1 The University of Iowa 2 Max Planck Institute for Intelligent Systems, Tubingen ¨ ¨ 3 Auburn University 1 muye@uiowa.edu 2 weiyang.liu@tuebingen.mpg.de 3 pan.he@auburn.edu ⇤ Corresponding Author https://vera-framework.github.io

## Abstract

The rapid advancement of vision-language models (VLMs) has established a new paradigm in video anomaly detection (VAD): leveraging VLMs to simultaneously detect anomalies and provide comprehendible explanations for the decisions. Existing work in this direction often assumes the complex reasoning required for VAD exceeds the capabilities of pretrained VLMs. Consequently, these approaches either incorporate specialized reasoning modules during inference or rely on instruction tuning datasets through additional training to adapt VLMs for VAD. However, such strategies often incur substantial computational costs or data annotation overhead. To address these challenges in explainable VAD, we introduce a verbalized learning framework named VERA that enables VLMs to perform VAD without model parameter modifications. Specifically, VERA automatically decomposes the complex reasoning required for VAD into reflections on simpler, more focused guiding questions capturing distinct abnormal patterns. It treats these reflective questions as learnable parameters and optimizes them through data-driven verbal interactions between learner and optimizer VLMs, using coarsely labeled training data. During inference, VERA embeds the learned questions into model prompts to guide VLMs in generating segment-level anomaly scores, which are then refined into frame-level scores via the fusion of scene and temporal contexts. Experimental results on challenging benchmarks demonstrate that the learned questions of VERA are highly adaptable, significantly improving both detection performance and explainability of VLMs for VAD.

## 1. Introduction

Video anomaly detection (VAD) aims to automatically identify unexpected and abnormal events in video sequences, with broad applications ranging from autonomous driving [2] to industrial manufacturing [31]. While achieving good performance in VAD is essential, providing clear explanations for detected anomalies is even more crucial.

To this end, our work primarily focuses on explain-

Figure 1. VERA renders frozen VLMs to describe and reason with learnable guiding questions learned from coarsely labeled data.

![Image](artifacts/image_000001_f6f6d340d1643b4c462777b50acf8b44f29669fcc6e8fb71dde433a03c0bc8d5.png)

able VAD, which requires both comprehensive visual understanding and the ability to generate human-interpretable predictions. The rapid advancement of vision-language models (VLMs) [7 , 18 , 21 , 58] enables us to address both requirements through their strong visual reasoning and language interaction capabilities. As multi-modal architectures that effectively combine the reasoning capabilities from large language models (LLMs) [4] and the visual understanding capabilities from pretrained vision encoders [8], VLMs are particularly well-suited for VAD for they can offer explainable predictions that clearly illustrate the rationale behind specific anomalies, making the results more interpretable to users. Recent research on VAD has consequently focused on how to effectively leverage the power of pretrained VLM. As shown in Fig. 1, existing approaches aim to address the misalignment problem between VLMs' pretraining tasks and the VAD requirements through either additional reasoning modules or instruction tuning (IT):

- One line of research introduces external LLMs to assist frozen VLMs to reason in VAD [46 , 52]. It uses VLMs to caption what they see given a video, and the descriptions are then passed to an external LLM, e.g., GPT-4 [1], to reason whether an anomaly occurs.
- Another line of research, instead, expands VLMs to generate explainable prediction via IT [26 , 55]. This research line creates additional VAD datasets with frame-

level annotations and leverages exemplary instructions to fine-tune the VLM, enabling it to detect anomalies and generate human-interpretable explanations.

Key Observations and Research Question. While prior research demonstrates the potential of applying VLMs to VAD, we identify that this new paradigm is hindered by a shared critical issue: the use of additional reasoning modules or fine-grained labeled datasets incurs significant computational cost either in the inference or training phases. First, decoupling a VAD system into a frozen VLM and an extra LLM introduces more overhead in inference, because it separates the description generation and reasoning processes. Secondly, although IT-based methods enable VLMs to effectively integrate description and reasoning for VAD, they require additional manpower and computational resources for annotating and finetuning on fine-grained labeled instruction datasets, which is time-consuming and not scalable for large-scale datasets. In light of this, we investigate the following unexplored yet important question:

Can we enable a frozen VLM to integrate description and reasoning for VAD without instruction tuning?

Our Approach. This research question is nontrivial because the reasoning ability of a frozen VLM is limited in general visual tasks, and it struggles to handle complex reasoning tasks like VAD, which requires the understanding of subtle, context-dependent outliers. To illustrate, Table 1 shows that prompting frozen VLMs with simple VAD questions used in existing works leads to unsatisfactory results. Thus, instruction-tuning a VLM seems necessary to make it responsive to specific instructional cues and capture delicate visual variations. In this paper, we question the necessity of such an operation and propose a principled approach to tailor frozen VLMs for VAD.

Specifically, our solution is guided by the intuition that the reasoning ability of VLMs for VAD will improve if we find questions with suitable and concrete description of abnormal patterns rather than with abstract and general words like "anomaly" to prompt them. Our idea is to iteratively refine anomaly descriptions from abstract ones (e.g., "is there any anomaly?") to detailed, specific characterizations.

Driven by such insight, we propose a framework, termed VERA, to explore verbalized learning (VL) for VAD. This framework considers the practical constraint that it is suboptimal to manually write down VAD guiding questions across VLMs, so it introduces a data-driven learning task to identify suitable anomaly-characterization questions containing concrete abnormal patterns for the frozen VLM using coarsely labeled datasets, eliminating the need for IT. Specifically, in the training phase, VERA treats the questions guiding the reasoning of VLMs in VAD as learnable parameters, improving them based on the verbal feedback from an optimizer VLM on the performance of a learner

| VAD Question for InternVL2-8B                       |   AUC (%) |
|-----------------------------------------------------|-----------|
| “Describe the video and is there any anomaly?” [26] |     53.05 |
| “Are there any abnormal events in the video?” [     |     65.03 |

Table 1. Instructing a frozen VLM (InternVL2-8B [7]) with simple questions to perform VAD yields poor AUC on UCF-Crime [32] dataset.

VLM on an intermediate VAD subtask—binary video classification for each video in the VAD training set. This design is both efficient and appropriate for VAD, as it accounts for video-specific properties like temporality while relying solely on provided coarse video-level labels. After that, considering the large scale of video frames, VERA assigns a fine-grained anomaly score for each frame in a coarse-tofine manner in the inference phase. First, VERA generates segment-level anomaly scores by querying VLMs with the learned guiding questions. Next, VERA improves the initial score by incorporating scene context into each segment score via ensembling. Finally, VERA outputs frame-level scores by fusing temporal context via Gaussian smoothing and frame-level position weighting.

Contributions. To sum up, our contributions are:

- To our knowledge, we present the first approach, that is, VERA, to adapt frozen VLMs as an integrated system for VAD by learning detailed anomaly-characterization questions in prompts that decompose anomalies into concrete and recognizable patterns. VERA learns them directly from coarsely labeled datasets, eliminating the need for IT or external reasoning modules.
- We introduce an effective VL-based algorithm for VLMs in VAD, allowing direct adaptation without modifying model parameters. With coarse labeled VAD datasets only, our approach obtains good guiding questions in VAD by relying on the verbal interaction between learner and optimizer VLMs in verbalized training. Additionally, we design a coarse-to-fine strategy to derive frame-level anomaly scores from verbally learned guiding questions in VAD, integrating both scene and temporal contexts for better VAD performance and reasoning.
- The learned guiding questions from VERA are expressed in natural languages, providing a unified method to encode and transfer prior VAD knowledge seamlessly to other datasets or VLMs. In challenging VAD datasets like UCF-Crime [32] and XD-Violence [42], VERA achieves state-of-the-art explainable VAD performance and enjoys good generalization ability across models and datasets.

## 2. Related Work

Video Anomaly Detection. VAD is the task of localizing frames that contain abnormal events in a given video. This task is challenging for anomalies cover a broad scope of events like accidents and criminal activities while training sets only offer coarse annotations. Modern VAD methods are based on deep neural networks (DNNs) for their superi-

ority and are going through a paradigm shift in using VLMs: (1) Early DNNs for VAD are task-specific, which often employ unsupervised (including one-class) or weakly supervised (WS) learning techniques for training. Most unsupervised learning methods [23 , 25 , 37 , 38 , 48 , 56] train DNNs on frame reconstruction/prediction tasks to establish representation spaces for normal/abnormal videos. WS learning methods [5 , 27 , 32 , 44 , 47 , 53] leverage both normal and abnormal videos to train a feature extractor that distinguishes anomalies from normalcy, typically using multiple instance learning [32] objectives. (2) Recent VAD methods adopt VLMs due to their remarkable success across core vision tasks [12 , 21 , 28 , 33]. Early research [26 , 46 , 52 , 55] has leveraged VLMs to generate textual descriptions of detected anomalies to enhance prediction explainability for VAD. However, current approaches incur high processing demands from external LLMs or require substantial effort and cost for fine-tuning on additional datasets, which are computationally inefficient in training or inference. Our work reduces the processing overhead by adapting frozen VLMs for VAD without model parameter modification or extra reasoning modules via learnable guiding questions, which elicit superior reasoning from frozen VLMs and significantly boost their performance in VAD.

Verbalized Learning for VLMs. The designed VL framework is inspired by a recent technique called verbalized machine learning (VML) [45]. The main idea of VML is to use LLMs to approximate functions and learn the verbal rules and descriptions of performing specific tasks, which casts traditional machine learning tasks as language-based learning tasks. This approach regards the language expressions that define classification rules and other task-specific criteria as learned parameters, and optimize them in a datadriven fashion through interactions between a learner and an optimizer modeled by LLMs or VLMs. However, the VML framework is limited to tasks involving regression on scalar values or classification for static images. A similar idea has also been explored in a concurrent method, TextGrad [49], which integrates the process of incorporating textual feedback from LLMs for improving prompts in PyTorch and further proves its effectiveness in coding, question answering, and optimization in chemistry and medicine. Compared to existing works, our work pioneers VL for the VAD task and video data, which remains unsolved for previous VL frameworks focus on static-data tasks and cannot handle the challenges of temporality and scene dynamics in videos. Specifically, VERA introduces a new learning paradigm for VAD: generating effective questions that encapsulate key abnormal patterns in videos to elicit the reasoning ability from VLMs for explainable VAD. Additionally, VERA works for any VAD dataset and supports WS learning. Unlike previous WS methods, VERA only needs to learn concise text but not millions of parameters, so the training is lightweight.

## 3. The VERA Framework

Our approach adapts VLMs to detect video anomalies without additional reasoning modules or IT. We now formulate the VAD task and detail the design of VERA.

## 3.1. Problem Formulation

Video Anomaly Detection. Let V be a video with F frames, represented as V = {Ii} F i=1 , where Ii is the i-th frame (1  i  F). Our objective is to locate and detect the start and end of anomalous events within V . In standard labeling, any frame associated with an anomaly is labeled as 1, and normal frames are labeled as 0. Therefore, the ground truth label sequence for V is Y = [y1,...,yF ], where yi 2 {0 , 1} represents the fine-grained label for Ii. We aim to use a frozen VLM, fVLM, to generate anomaly score predictions across all frames, Y ˆ = [ˆy1 ,..., y ˆ F ], where y ˆ i 2 [0 , 1] is a continuous anomaly score for Ii .

Available Training Data for VAD. Typically, VAD datasets only provide coarsely labeled training sets [23 , 25 , 32 , 42]. We denote a VAD training set as D = {(V (j), Y (j))} N j=1 , where N is the total number of training videos, V (j) represents the j-th video (1  j  N) and Y (j) is the corresponding video-level label. Y (j) = 1 if V (j) contains any anomaly defined by the dataset annotators, e.g., abuse or arson activities, and Y (j) = 0 if V (j) has no anomalies. For V (j), we suppose it contains Fj Fj frames and denote the frames sequence as V (j) = {I
i (j) I
i } Fj Fj i=1 , where I
i (j) I
i is the i-th frame (1  i  Fj ) in V (j) .

## 3.2. Training in VERA

Training Objective. We aim to learn guiding questions that break down a complex and ambiguous concept (i.e., what is an "anomaly") into a set of identifiable anomalous patterns to unlock reasoning capabilities within frozen VLMs for VAD tasks. Those patterns vary among datasets, making manually designed descriptions ineffective for generalization. To address this, we propose a general VL framework shown in Fig. 2 to generate the desired guiding questions. We denote the guiding question set as Q = {q1,...,q m } , where qi is the i-th question (1  i  m) and m is the number of questions. The training framework considers Q as the learnable parameters, which are optimized through verbal interaction between a learner and an optimizer, modeled by VLMs through leveraging their ability to follow instructions with given prompts.

Training Data. The training data for learning Q consist of paired sampled video frames and video-level labels. Sampling is necessary because the amount of video frames is so huge that we cannot compute with every frame. We explore three types of sampling strategies and find that uniform sampling [54] yields the best results. That is, with any video V (j) 2 D, we first calculate the interval between

Figure 2. The overall training pipeline in VERA aims to optimize VAD guiding questions iteratively. In each iteration t, the optimization is verbalized by providing verbal instructions for the learner and optimizer to follow. They will generate predictions and new guiding questions, respectively.

![Image](artifacts/image_000002_7d4b19a8f86a4e765e4ede218990d8f7b25c2d4e27531c0e002a7475a7f9e315.png)

sampled frames as l = floor(Fj/S), where S is the number of sampled frames, and floor denotes rounding down to the nearest integer. Given l, the uniformly sampled frames from V (j) are represented by V˜ ˜ (j) = [I (j) 1 , I 
l (j) I 
l+1 ,...,I (j) (S1)·l+1 ] . The label used for training is Y (j) only, resulting in training data pairs {(V ˜ (j) , Y (j))} N j=1 for VERA.

Updating Q via Learner and Optimizer. Since Q are verbal expressions for specific anomaly patterns, VERA inherits the idea of VML [45] in training: optimizing language-based parameters by verbal communication between a learner agent flearner and an optimizer agent fopt, rather than by numerical optimization algorithms like Adam [16]. W.l.o.g., we take an arbitrary iteration t when implementing the complete algorithm (detailed in Supplementary Material) for illustration. We denote any LLMbased model as f(x; ) where x represents the input data, and  denotes the natural language instructions for f to follow, which is considered as learnable parameters in our VL framework. Specifically, Q contains parameters to be learned in VERA. As depicted in Fig. 2, in each iteration t, the learner agent f
l (t) f
learner is modeled by the frozen VLM fVLM(·) used for VAD with a specific prompt template ✓ that guide fVLM(·) to conduct a learning task by pondering on current guiding questions Qt. We denote the learner agent as f
l (t) f
learner (x) = fVLM(x; (✓ , Qt)), where x is the input in a learning task, and Qt, the learnable guiding questions applied in each iteration t, constitutes the core parameters that distinguish the learner between iterations. Meanwhile, we introduce an optimizer f
o (t) f
opt to assess the quality of the

predictions of the learner and to optimize Qt. W.l.o.g., we use the same frozen VLM fVLM to model the optimizer. As demonstrated in Fig. 2, we provide another specific prompt template for the learner to follow to optimize Qt, so we denote the optimizer agent as f
o (t) f
opt (z) = fVLM(z; ( , Qt)) , where z is its input and is the instruction to improve Qt . It is important to note that f
l (t) f
learner 6= f
o (t) f
opt because f
l (t) f
learner follows (✓ , Qt) to conduct a learning task, while f
o (t) f
opt follows ( , Qt) to refine Qt .

Learning Task for flearner. The learner executes the "forward pass" and outputs a prediction. Recall that we only use the original coarsely labeled information for training. Thus, we design a binary classification task for flearner, which accounts for the temporal nature of video data, the sparsity of anomalies, and the weak supervision in VAD datasets. In this task, the job of the learner flearner is to produce a binary classification prediction Y ˆ (j) to determine whether there is an anomaly in the video based on the sampled frames V ˜ (j) . As shown in Fig. 2, we explain the task in natural language in the "Model Description" section in ✓. Guiding questions Qt are inserted in the "Prompt Questions" section in ✓ to elicit reasoning of the VLM. This template design is based on the prompt structures used in VML, with targeted modifications to help the learner effectively address this WS learning task. Given ✓ and a sampled frame set V ˜ (j)
, the learner will output a prediction as

<!-- formula-not-decoded -->

where Y ˆ (j) = 1 if the learner thinks there is an anomaly af-

ter skimming across the sampled frames V ˜ (j) and reasoning through the guiding questions Qt, and otherwise, Y
i ˆ Y
i = 0 . Optimization Step in fopt. The optimizer executes the "backward pass" to update the questions Qt via a mini-batch (batch size is n). Suppose the visual input in a batch is Vbatch = [V
b ˜ (1) V
batch, ··· , V
b ˜ (n) V
batch ] and the corresponding ground truths are Ybatch = [Y 
b (1) Y 
batch, ··· , Y 
b (n) Y 
batch ]. The learner generates prediction as Y
b ˆ Y
batch = [Y
b ˆ (1) Y
batch, ··· , Y
b ˆ (n) Y
batch ] with the current questions Qt by Eq. (1). The optimizer will output a new set of questions Qt+1 by following the prompt with batched data. We denote the optimization step as

<!-- formula-not-decoded -->

where Qt+1 is a new set of guiding questions constructed from f
o (t) f
opt owing to its text generation and instruction following abilities after reading .

## 3.3. Inference in VERA

During training, we denote the one with the largest validation accuracy as Q ⇤ . In inference, given Q ⇤ , VERA yields fine-grained anomaly score Y ˆ for a test video V via a coarse-to-fine process shown in Fig. 3 .

Step 1: Initial Anomaly Scores via Learned Guiding Questions. We divide the video into segments and analyze each segment independently first. Following [52], we perform equidistant frame sampling within V to obtain the set of segment centers C = {I1, Id+1 , ··· , I(h 1)·d+1 }, where d is the interval between centers and h = floor(F/d) is the total number of segments. For each center frame I( u 1)·d+1 (1  u  h), we define a 10-second window around it as the u -th segment, within which we uniformly sample 8 frames. We denote the sampled frame set in the u-th segment as Vu Vu. Next, we input Vu Vu in fVLM with the prompt (✓ , Q ⇤ ) to get the initial score

<!-- formula-not-decoded -->

where y˜ ˜ u = 1 if fVLM thinks the segment contains an anomaly after reasoning via Q ⇤ with Vu Vu , and otherwise, y ˜ u = 0. By repeating Eq. (3) for each segment, we have a segment-level initial anomaly score set Y ˜ = [˜y1 , ··· , y ˜ h ] . Step 2: Ensemble Segment-Level Anomaly Scores with Scene Context. Note that the scores derived above only examine a short moment in a long video without considering any context. To resolve it, we refine the initial segment-level score by incorporating scene context—defined as preceding and following segments that contain similar elements, such as actors and background, to those in the current segment.

We measure the relevance between different video segments by the cosine similarity of their feature representations [22], extracted by a pretrained vision feature extractor

Step 1: Initial Anomaly Scores via Learned Guiding Questions

![Image](artifacts/image_000003_cf37e2f632465dd4e8ea3691d3caaba7ebce4ce004c565367c0c069f275fcfd9.png)

Figure 3. VERA computes anomaly scores with Q ⇤ in three steps.

g , e.g., ImageBind [10]. For the u-th segment Vu Vu , its similarity with any segment Vw Vw (1  w  h) is sim(u, w) = cos ⇣ e u · e w ||e u ||·||e w || ⌘
, where cos denotes the cosine function, and e u = g(Vu Vu ) and e w = g(Vw Vw ) represent their features. Let  u = [ (1) u ,...,  (K) u ] denote the indices of the top-K segments similar to Vu Vu . We refine the anomaly score by

<!-- formula-not-decoded -->

where y¯ ¯ u is an ensemble of initial scores of top-K video segments relevant to Vu Vu . Here, the initial score of each retrieved segment is weighted by a factor derived from the cosine similarity and normalized by the Softmax function (with ⌧ as the temperature hyperparameter). Accordingly, scenes with greater similarity are assigned higher weights, making the ensemble score a more comprehensive reflection of anomalies with the video context. By applying Eq. (4) for all segments, we obtain Y ¯ = [¯y1 ,..., y ¯ h ] .

Step 3: Frame-level Anomaly Scoring with Temporal Context. Given Y ¯ , we aim to incorporate temporal context to capture how events evolve over time when computing frame-level anomaly scores, for the abnormality of an event often depends on the timing and progression of observed activities. To detail, we first apply Gaussian smoothing [11] to aggregate local temporal context into the segment-level anomaly scores. We denote the Gaussian kernel (suppose the filter size is !) as G(p) = exp( p 2 2 2 1 ) where p is the distance from the kernel center and 1 is the variance. We update segment-level scores as ¯ = Y ¯ ⇤ G = [¯1 , ··· , ¯ h ] , where ⇤ is the convolution operation. Next, we integrate global temporal context by position weighting. With ¯ , we flatten it into frame-level scores by assigning the score ¯ ¯ u to each frame in the u-th segment, i.e ., [I( u 1)·d+1, ··· , Iu Iu· d ] . We denote the frame-level score sequence after flattening as [⇢1 , ··· , ⇢F ]. We then apply the Gaussian function to encode position weights as w(i) = exp ⇣ (ic) 2 2 2 2 ⌘
, where

i (1  i  F) is any frame index, c = floor(F/2) is the center frame index, and 2 is the variance. The anomaly score for the i-th frame is:

<!-- formula-not-decoded -->

This operation scales the score ⇢i, diminishing the anomaly score for frames near the beginning and end of the event. This helps better capture the temporal progression of anomalies: the score gradually increases as the anomaly reaches its peak and decreases afterward. The final scores is denoted as Y ˆ = [ˆy1 ,..., y ˆ F ] after applying Eq. (5).

Explainable VAD by VERA. When using template ✓ embedded with Q ⇤ to compute Y ˆ , we ask the VLM to "provide an explanation in one sentence" when reasoning, and VLM will explain the anomaly score it assigns based on Q ⇤ .

## 4. Experiments and Results

In this section, we present an evaluation of VERA as follows, addressing key questions of interest including: (Q1) Does it enhance the effectiveness of frozen VLMs in VAD? (Q2) Is its design reasonable and well-structured? (Q3) How well does it generalize across different scenarios?

## 4.1. Experimental Settings

Datasets. We conduct experiments on two large-scale VAD datasets: (1) UCF-Crime [32] collected from surveillance videos with 13 types of anomalies and 290 (140 abnormal) test videos (2.13 minutes long on average). (2) XDViolence [42] with 6 anomaly categories and 800 (500 abnormal) test videos (1.62 minutes long on average).

Metrics. Following approaches in [52 , 55], we mainly evaluate VAD performance using the Area Under the Curve (AUC) of the frame-level Receiver Operating Characteristic (ROC) curve, as it provides a comprehensive measure of model performance across all thresholds.

Baselines. We categorize baselines into non-explainable approaches and explainable ones as [55] does. Nonexplainable ones are obtained by WS learning [6 , 9 , 15 , 17 , 19 , 32 , 36 , 41 – 43 , 50 , 51 , 57] and unsupervised learning [13 , 25 , 34 , 35 , 37 , 38]. These non-explainable approaches cannot provide language-based explanations for VAD. For explainable approaches, we use LAVAD [52], Holmes-VAD [55], and VADor [26] as representatives of Pipeline 1 and Pipeline 2 shown in Fig. 1. It should be noted that [46] does not report performance on UCF-Crime and XD-Violence. Additionally, we include zero-shot (ZS) VAD by frozen VLMs designed by [52] as baselines.

Implementation of VERA. In our experiments, we choose a small VLM, InternVL2-8B [7], as the backbone fVLM for building VERA by default, if not otherwise specified. We also explore other backbones, such as Qwen2-VL-7B [40] and InternVL2-40B [7] for ablation. We train Q for no more than 10 epochs, with a validation accuracy calculated every 100 iterations to determine Q ⇤ . We set n as 2, S as 8, and m as 5 for training. The initial questions Q0 is "1. Is there any suspicious person or object that looks unusual in this scene? 2. Is there any behavior that looks unusual in this scene?", inspired by previous VAD methods [13 , 43], which assume anomalies appear with unusual appearance or motions.

## 4.2. Comparison to State-of-the-art Methods

We address Q1 by empirically comparing VERA to existing VAD methods. First, in Table 2, VERA achieves the highest AUC among explainable VAD methods on UCF-Crime, outperforming Holmes-VAD and VADor (without IT, as reported in their papers) in a fair comparison. Importantly, unlike these methods, VERA does not need to modify the model parameters, demonstrating its suitability to directly adapt VLM to the VAD task with minimal training requirements. Moreover, VERA surpasses LAVAD by 6% in AUC on UCF-Crime, uniquely integrating both description and reasoning capabilities in VAD. Compared to non-explainable methods, VERA achieves AUC performance that is comparable to one of the top-performing

Table 2. AUC (%) on UCF-Crime. No IT is used for Holmes-VAD and VADor.

| Method                      | AUC                     |
|-----------------------------|-------------------------|
| Non-explainable VAD Methods | Methods                 |
| Wu et al. [42]              | 82.44                   |
| OVVAD [43]                  | 86.40                   |
| S3R [41]                    | 85.99                   |
| RTFM [36]                   | 84.30                   |
| MSL [19]                    | 85.62                   |
| MGFN [6]                    | 86.98                   |
| SSRL [17]                   | 87.43                   |
| CLIP-TSA [15]               | 87.58                   |
| Sultani et al. [32]         | 77.92                   |
| GCL [51]                    | 79.84                   |
| GCN [57]                    | 82.12                   |
| MIST [9]                    | 82.30                   |
| CLAWS [50]                  | 83.03                   |
| DYANNET [35]                | 84.50                   |
| Tur el al. [37]             | 66.85                   |
| GODS [38]                   | 70.46                   |
| Explainable VAD Methods     | Explainable VAD Methods |
| LAVAD [52]                  | 80.28                   |
| Holmes-VAD [55]             | 84.61                   |
| VADor [26]                  | 85.90                   |
| ZS CLIP [52]                | 53.16                   |
| ZS IMAGEBIND-I [52]         | 53.65                   |
| ZS IMAGEBIND-V [52]         | 55.78                   |
| LLAVA-1.5 [20]              | 72.84                   |
| VERA                        | 86.55                   |

methods, CLIP-TSA, on UCF-Crime, while offering the additional advantage of explainable predictions.

Similar advantages are also observed in Table 3 for XD-Violence. Considering multiple factors, including performance, training efficiency, system integration, and explainability, VERA stands out as a promising pipeline for VLMs in VAD.

## 4.3. Ablation Studies

We perform necessary ablation studies on UCF-Crime to answer both Q2 and Q3

Table 3. AUC (%) on XD-Violence.

| Method                      | AUC                         |
|-----------------------------|-----------------------------|
| Non-Explainable VAD Methods | Non-Explainable VAD Methods |
| Hasan et al. [13]           | 50.32                       |
| Lu et al. [25]              | 53.56                       |
| BODS [38]                   | 57.32                       |
| GODS [38]                   | 61.56                       |
| RareAnom [34]               | 68.33                       |
| Explainable VAD Methods     | Explainable VAD Methods     |
| LAVAD [52]                  | 85.36                       |
| ZS CLIP [52]                | 38.21                       |
| ZS IMAGEBIND-I [52]         | 58.81                       |
| ZS IMAGEBIND-V [52]         | 55.06                       |
| LLAVA-1.5 [20]              | 79.62                       |
| VERA                        | 88.26                       |

for a comprehensive evaluation on our design choices.

Frame Sampling Strategy in Training. We compare three frame sampling strategies for obtaining each V ˜ (j) in training: uniform sampling, random sampling, and TSN sampling (random sampling from equally divided segments).

Table 4 shows that uniform sampling performs the best (with batch size n = 2 and S = 8). This is because uniform sampling preserves the temporal structure and maintains consistent motion patterns throughout

Table 4. Sampling strategies explored in VERA training.

| Strategy     |   AUC (%) |
|--------------|-----------|
| Random [3]   |     83.63 |
| TSN [39]     |     82.63 |
| Uniform [54] |     86.55 |

the long video, making it easier for VLMs to understand the video and update Q .

Table 5. The way we obtain guiding questions affects AUC substantially.

| Question Type                                                 |   AUC (%) |
|---------------------------------------------------------------|-----------|
| No questions                                                  |     78.81 |
| Manually written questions by human                           |     81.15 |
| Learned questions w/o iteratively inputting Vbatch in Eq. (2) |     78.06 |
| Iteratively learned questions (used in VERA)                  |     86.55 |

How to Obtain Guiding Questions Q for VLM. As seen in Table 5, if the guiding questions are not incorporated into the VLM prompt, the AUC will drop largely to 78.81%, confirming the need to use simpler and more focused questions to provoke reasoning in the VLMs for VAD. Meanwhile, if we use manually written questions (Q0), the performance is suboptimal with an 81.15% AUC, which shows the need to use VL to find guiding questions. Lastly, if we only input batched predictions Y
b ˆ Y
batch and ground truths Yb Ybatch without inputting Vbatch in the optimizer, the Q updated in this way will dumb the VLMs and make it have a low AUC. Thus, inputting video frames as Eq. (2) does is necessary to learn good Q .

Number of Questions m . As shown in Fig. 4, when m is set to 1, the reasoning is limited to a single perspective, resulting in a lower AUC. As m increases up to 5, the model captures more comprehensive anomaly patterns, leading to improved AUC. However, increasing m

Figure 4. Effect of the number of guiding questions on AUC.

![Image](artifacts/image_000004_15c773950227e7b8ea57bb945dde1a52fa1ffb85249802dab31169c7f2c4d042.png)

beyond 5 yields no significant gains. Therefore, we set m to 5 by default in VERA, if not otherwise specified.

Table 6. Ablation study of each step in VERA’s inference.

| Operation                                            | AUC (%)       |
|------------------------------------------------------|---------------|
| Initial (Step 1)                                     | 76.10         |
| Initial + Retrieval (Step 2)                         | 84.53 (+8.43) |
| Initial + Retrieval + Smoothing (Step 3)             | 85.48 (+0.95) |
| Initial + Retrieval + Smoothing + Weighting (Step 3) | 86.55 (+1.07) |

Coarse-to-Fine Anomaly Score Computation. We also validate the anomaly score computation by VERA. Table 6

shows the AUC is 76.10% when using the flattened initial score obtained in Step 1, and leveraging retrieved segments in Step 2 significantly boosts the AUC to 84.53%, highlighting the effectiveness of incorporating ensemble scores based on scene context. Meanwhile, smoothing and weighting in Step 3 further improves the AUC by around 1% each, verifying the benefit of integrating temporal context.

Generalizability Test. We further examine the generalizability of VERA across different model sizes, VLM architectures, and datasets to address Q3.

First, we apply VERA to InternVL2-40B, a larger model in the InternVL2 family compared to InternVL2-8B. As shown in Table 7, InternVL2-40B achieves effective AUC performance, slightly exceeding that of InternVL2-8B, indicating that VL in VERA enables models of various scales to identify a Q suitable

Table 9. AUC (%) across datasets.

| fVLM                  | Source of Q                        | Source of Q                        |
|-----------------------|------------------------------------|------------------------------------|
| fVLM                  | InternVL2-8B                       | 8B InternVL2-40B                   |
| InternVL2-8B          | 86.55                              | 80.43                              |
| InternVL2-40B         | 85.24                              | 86.72                              |
| Table 7. AUC (%) acro | AUC (%) across model sizes .       | AUC (%) across model sizes .       |
| fVLM                  | Source of Q
 InternVL28B Qwen2VL7B | Source of Q
 InternVL28B Qwen2VL7B |
| InternVL2-8B          | 86.55                              | 81.37                              |
| Qwen2-VL-7B           | 79.60                              | 82.64                              |
| Table 8. AUC (%) a    | UC (%) across architectures.       | UC (%) across architectures.       |
| Dataset               | Source of Q                        | Source of Q                        |
|                       | UCF-Crime X                        | e XD-Violence                      |
| UCF-Crime             | 86.55                              | 88.26                              |
| XD-Violence           | 86.26                              | 88.26                              |

for their reasoning capabilities. Additionally, We also evaluate the transferability of Q across different scales and and observe an interesting phenomenon: the Q learned by InternVL2-8B remains effective for InternVL2-40B, but not vice versa. This is likely because the Q learned by the smaller model is readily interpretable by the larger model, whereas the Q derived from the larger model is more complex in syntactic structure and does not align well with the reasoning framework of the smaller model. Secondly, we select a different VLM, Qwen2-VL-7B [40], as the backbone for VERA. As shown in Table 8, while the AUC achieved with Qwen2-VL-7B is lower than that with InternVL2-8B, the VL in VERA remains effective, allowing it to outperform notable baselines such as LAVAD [52]. However, a notable gap exists when transferring Q across different model architectures in Table 8. Developing a universal Q that can effectively elicit reasoning capabilities across various VLM structures would be an promising direction for future research. Lastly, we observe that the transferability of Q depends on the training dataset. From Table 9, we observe that transferring Q learned from UCFCrime to XD-Violence results in a smaller performance drop compared to the reverse case. This suggests the source dataset is crucial to the transferability of Q across datasets.

## 4.4. Qualitative Results and Case Studies

W.l.o.g., we take one video on UCF-Crime to illustrate the explainability brought by the learned Q ⇤ qualitatively (on

1

Learned Guiding Questions !

∗

in VERA

.

Are there any people in the video who are not in their typical positions or engaging in

activities that are not consistent with their usual behavior?

2

.

Are there any vehicles in the video that are not in their typical positions or being used

in a way that is not consistent with their usual function?

3

.

Are there any objects in the video that are not in their typical positions or being used

in a way that is not consistent with their usual function?

4

.

Is there any visible damage or unusual movement

in the video that indicates an

anomaly?

5

.

Are there any unusual sounds or noises in the video that suggest an anomaly?

Figure 5. Given Q ⇤ by VERA, the frozen VLM (InternVL2-8B) will reason and explain the scene based on it. For illustration, we take as an example the video "Arrest007 x264" from UCF-Crime and include 6 scenes here. The complete anomaly scores are shown in Fig. 8 .

![Image](artifacts/image_000005_196c39891e3c6c3dba405f63b3bf35832ab2e070f930e5a7679e28a7a0fab2d2.png)

UCF-Crime Q ⇤ is "1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior? 2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function? 3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function? 4. Is there any visible damage or unusual movement in the video that indicates an anomaly? 5. Are there any unusual sounds or noises in the video that suggest an anomaly?"). As shown in Fig. 5, the main anomaly in this video is that a man tries to steal money from the washing machines in a laundromat and is arrested after being found by the police. In the selected 6 main video segments, the frozen VLM with VERA's learned questions is able to explain the scene by closely following the detailed anomaly characterization of the five learned guiding questions. W.l.o.g., we take the first 3 segments in Fig. 5 for instance to closely compare the caption quality with LAVAD, a representative baseline. As shown in Fig. 6, VERA's captions include both precise descriptions (bold text) and reasoning (text in purple) about anomalies, while LAVAD's captions contain only plain descriptions. This difference owes to VERA's learned guiding questions, which transform VLM's thinking and phrasing.

A more interesting advantage of VERA is that it allows humans to further interact with VLMs because it retains the general question-answering ability of pretrained VLMs. This is because VERA does not require finetuning of the VLM backbone weights. Although finetuning VLMs with parameter-efficient methods like [14 , 24 , 29] is easy and computationally tractable, instruction-tuned models still inevitably lose the flexibility to handle general questions (due to catastrophic forgetting), as they are trained to respond to certain queries with fixed answer styles. In contrast, as shown in Fig. 7, the learned Q ⇤ can steer reasoning in a frozen VLM while allowing it to flexibly answer openended (like follow-up or counterfactual) questions, which is

Figure 6. Qualitative comparison between VERA and LAVAD.

![Image](artifacts/image_000006_b9fae0c3ec06aa49651e08233ad1d567115c3bc1d6722b1586e7d6233c7240f4.png)

Figure 7. VERA can take open-ended questions and interact with humans.

![Image](artifacts/image_000007_584091d3c688abac2dc99208f70c9f92185aa1b37ce3ef0be3e2c84517e78bb9.png)

an important ability lost in IT-based models.

Moreover, as shown in Fig. 8, owing to the proposed coarse-to-fine anomaly scoring, the anomaly score dynamics from VERA well represent the actual real-time anomaly level in this video and gradually increases to nearly 1 when the man is being arrested. This result verifies that VERA allows VLMs to effec-

Figure 8. Anomaly scores generated by VERA (with InternVL2-8B) in "Arrest007 x264" from UCF-Crime.

![Image](artifacts/image_000008_972ff40ca1667c6cba40f6af195d664c4c0f40c42ba8f73e554786d349bc4c9d.png)

tively identify anomalies with a holistic model, reducing the manpower and computational overhead for VAD.

## 5. Concluding Remarks

We propose a novel pipeline, VERA, which can effectively elicit the reasoning ability from VLMs to perform explainable VAD without additional computation overhead. This is done through an effective and novel application of verbalized machine learning [45] to VLM. In training, VERA obtains the guiding questions detailing anomaly patterns through the verbal interaction between the learner and the optimizer agents. In inference, VERA uses them to enhance VLMs for identifying anomalies and compute frame-level anomaly scores in a coarse-to-fine process. Experimental results validate the effectiveness of the VERA framework in achieving state-of-the-art explainable VAD performance.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023. 1
- [2] Daniel Bogdoll, Maximilian Nitsche, and J Marius Zollner. ¨ ¨ Anomaly detection in autonomous driving: A survey. In CVPR Workshops, 2022. 1
- [3] Meinardus Boris, Batra Anil, Rohrbach Anna, and Rohrbach Marcus. The surprising effectiveness of multimodal large language models for video moment retrieval. arXiv preprint arXiv:2406.18113, 2024. 7
- [4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. 1
- [5] Junxi Chen, Liang Li, Li Su, Zheng-jun Zha, and Qingming Huang. Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection. In CVPR , 2024. 3
- [6] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In AAAI, 2023. 6 , 14
- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR , 2024. 1 , 2 , 6
- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 1
- [9] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In CVPR, 2021. 6
- [10] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In CVPR, 2023. 5 , 15
- [11] Rafael C Gonzalez. Digital image processing. Pearson education india, 2009. 5
- [12] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In CVPR, 2024. 3
- [13] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K

Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In CVPR, 2016. 6 , 11

- [14] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2021. 8
- [15] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection. In ICIP, 2023. 6 , 14
- [16] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 4
- [17] Guoqiu Li, Guanxiong Cai, Xingyu Zeng, and Rui Zhao. Scale-aware spatio-temporal relation learning for video anomaly detection. In ECCV, 2022. 6
- [18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML , 2023. 1
- [19] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. In AAAI, 2022. 6 , 14
- [20] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR , 2024. 6 , 14
- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 1 , 3
- [22] Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. Deep hyperspherical learning. In NeurIPS, 2017. 5
- [23] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In CVPR, 2018. 3
- [24] Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, et al. Parameter-efficient orthogonal finetuning via butterfly factorization. In ICLR, 2024. 8
- [25] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In ICCV, 2013. 3 , 6
- [26] Hui Lv and Qianru Sun. Video anomaly detection and explanation via large language models. arXiv preprint arXiv:2401.05702, 2024. 1 , 2 , 3 , 6
- [27] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In CVPR, 2023. 3
- [28] Sarah Pratt, Ian Covert, Rosanne Liu, and Ali Farhadi. What does a platypus look like? generating customized prompts for zero-shot image classification. In ICCV, 2023. 3
- [29] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Scholkopf. Controlling text-to-image diffusion by orthogo- ¨ ¨ nal finetuning. In NeurIPS, 2023. 8
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 18

- [31] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Scholkopf, Thomas Brox, and Peter Gehler. Towards total ¨ ¨ recall in industrial anomaly detection. In CVPR, 2022. 1
- [32] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In CVPR, 2018. 2 , 3 , 6 , 14
- [33] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Yingcong Chen. Hawk: Learning to understand open-world video anomalies. Advances in Neural Information Processing Systems, 37:139751–139785, 2024. 3
- [34] Kamalakar Vijay Thakare, Debi Prosad Dogra, Heeseung Choi, Haksub Kim, and Ig-Jae Kim. Rareanom: A benchmark video dataset for rare type anomalies. Pattern Recognition, 140:109567, 2023. 6
- [35] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. Dyannet: A scene dynamicity guided self-trained video anomaly detection network. In WACV, 2023. 6
- [36] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In ICCV, 2021. 6 , 14
- [37] Anil Osman Tur, Nicola Dall'Asen, Cigdem Beyan, and Elisa Ricci. Unsupervised video anomaly detection with diffusion models conditioned on compact motion representations. In International Conference on Image Analysis and Processing, 2023. 3 , 6
- [38] Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In ICCV, V, 2019. 3 , 6
- [39] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In ECCV, 2016. 7
- [40] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6 , 7
- [41] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Self-supervised sparse representation for video anomaly detection. In ECCV, 2022. 6 , 14
- [42] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In ECCV, 2020. 2 , 3 , 6 , 14
- [43] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In CVPR, pages 18297–18307, 2024. 6 , 11 , 14
- [44] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6074–6082, 2024. 3
- [45] Tim Z Xiao, Robert Bamler, Bernhard Scholkopf, and ¨ ¨ Weiyang Liu. Verbalized machine learning: Revisiting machine learning with language models. arXiv preprint arXiv:2406.04344, 2024. 3 , 4 , 8
- [46] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: reasoning for video anomaly detection with large language models. arXiv preprint arXiv:2407.10299, 2024. 1 , 3 , 6
- [47] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. In CVPR, 2024. 3
- [48] Muchao Ye, Xiaojiang Peng, Weihao Gan, Wei Wu, and Yu Qiao. Anopcn: Video anomaly detection via deep predictive coding network. In ACM international conference on multimedia, 2019. 3
- [49] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic "differentiation" via text. arXiv preprint arXiv:2406.07496, 2024. 3
- [50] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In ECCV, 2020. 6
- [51] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In CVPR, 2022. 6
- [52] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In CVPR , 2024. 1 , 3 , 5 , 6 , 7 , 14 , 15
- [53] Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang. Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection. In CVPR, 2023. 3
- [54] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In EMNLP, 2023. 3 , 7
- [55] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao, Yuehuan Wang, and Nong Sang. Holmes-vad: Towards unbiased and explainable video anomaly detection via multi-modal llm. arXiv preprint arXiv:2406.12235, 2024. 1 , 2 , 3 , 6 , 14
- [56] Menghao Zhang, Jingyu Wang, Qi Qi, Haifeng Sun, Zirui Zhuang, Pengfei Ren, Ruilong Ma, and Jianxin Liao. Multiscale video anomaly detection by multi-grained spatiotemporal representation learning. In CVPR, 2024. 3
- [57] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In CVPR, 2019. 6
- [58] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024. 1