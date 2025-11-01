---
title: 'Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal
  LLM'
type: method
categories:
- Hybrid
description: A novel framework leveraging multimodal instructions and 
  large-scale datasets to enable unbiased, interpretable, and accurate video 
  anomaly detection with large language models, including a new dataset 
  VAD-Instruct50k with single-frame annotations and explanatory instruction 
  data.
benchmarks:
- shanghaitech
- ucf-crime
- xd-violence
authors:
- Huaxin Zhang
- Xiaohao Xu
- Xiang Wang
- Jialong Zuo
- Chuchu Han
- Xiaonan Huang
- Changxin Gao
- Yuehuan Wang
- Nong Sang
date: '2023-10-01'
---

## Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal LLM

Huaxin Zhang 1 , 3 , Xiaohao Xu 2 , Xiang Wang 1 , Jialong Zuo 1 , Chuchu Han 1 , 3 ,

Xiaonan Huang 2 , Changxin Gao 1 , Yuehuan Wang 1 , Nong Sang 1

1 Key Laboratory of Image Processing and Intelligent Control, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

2 University of Michigan, Ann Arbor 3 Baidu Inc. Corresponding Author

## Abstract

Towards open-ended Video Anomaly Detection (VAD), existing methods often exhibit biased detection when faced with challenging or unseen events and lack interpretability. To address these drawbacks, we propose Holmes-VAD, a novel framework that leverages precise temporal supervision and rich multimodal instructions to enable accurate anomaly localization and comprehensive explanations. Firstly, towards unbiased and explainable VAD system, we construct the first largescale multimodal VAD instruction-tuning benchmark, i.e. , VAD-Instruct50k. This dataset is created using a carefully designed semi-automatic labeling paradigm. Efficient single-frame annotations are applied to the collected untrimmed videos, which are then synthesized into high-quality analyses of both abnormal and normal video clips using a robust off-the-shelf video captioner and a large language model (LLM). Building upon the VAD-Instruct50k dataset, we develop a customized solution for interpretable video anomaly detection. We train a lightweight temporal sampler to select frames with high anomaly response and fine-tune a multimodal large language model (LLM) to generate explanatory content. Extensive experimental results validate the generality and interpretability of the proposed Holmes-VAD , establishing it as a novel interpretable technique for real-world video anomaly analysis. To support the community, our benchmark and model will be publicly available at https://holmesvad.github.io/ .

## 1 Introduction

Video Anomaly Detection (VAD) [14] aims to identify abnormal events in videos, which has been extensively researched in recent years due to its considerable application value in public safety [43] and video content understanding [55]. Current VAD approaches can be broadly classified into three categories according to the annotation type of the training data, i . e ., unsupervised, weaklysupervised and fully-supervised. Unsupervised methods [14 , 35 , 30 , 12 , 49 , 60] train solely on normal videos (one-class) or unlabeled normal/abnormal videos, while weakly supervised methods [43 , 71 , 45 , 23 , 54 , 72 , 37] train on normal/abnormal videos with video-level labels. Fully-supervised methods [29 , 19] are less studied due to the high cost of precise frame-by-frame annotations. Recently, inspired by the strong representation of multi-modal large language models (MLLMs) pretrained on massive data [46 , 7 , 16 , 59 , 26 , 74 , 28 , 69 , 62 , 9 , 4 , 51] and their impressive advancements in many downstream visual tasks [13 , 52 , 53], many efforts [41 , 17 , 57 , 61 , 56 , 65] start to integrate the multi-modal knowledge into VAD systems, which enables more precise anomaly detection.

Despite significant progress, existing VAD models still face the following primary challenges:

Preprint. Under review.

Figure 1: Towards unbiased and explainable VAD. In contrast to prevailing VAD approaches (a) that primarily concentrate on identifying anomalies, our method (b) facilitates not only unbiased (i.e., less false alarms toward easily cofused or unseen normality) predictions of anomaly scores but also explanation of detected anomalies, through constructing a large scale VAD dataset with single-frame annotations for untrimmed videos and explanable instruction data for trimmed videos.

![Image](artifacts/image_000000_42e06dfc5c044f9996feab15bec3e4b412f1e4f0ef47e164fbd2cb805a5f43a4.png)

- Biased anomaly space: Due to the lack of reliable frame-level abnormal supervision, unsupervised methods fail to reconstruct or predict unseen normal data, while weakly supervised methods also struggle to select trustworthy snippets for training under the videolevel supervision. Consequently, the learned anomaly space of these methods develop a prevalent bias toward unseen or easily-confused normality, remain "when does the anomaly happen" still facing challenges. Although there are some fully supervised studies [29 , 19], the number of annotated videos is very small due to the inefficiency of the annotation process, resulting in a lack of scalability.
- Lack of explainability: Existing video anomaly detection approaches do not offer transparent explanations and reasoning for detected anomalies, i . e ., "what is the anomaly" , and "why is it considered anomalous". This opacity restricts human comprehension and engagement with the system.

Drawing from the above analysis, our insight is that a strong AI-powered anomaly detection system requires not only identifying deviations, but also providing insightful explanations, mirroring the deductive reasoning like the detective Sherlock Holmes. To this end, we present Holmes-VAD, an unbiased and explainable VAD framework based on MLLMs (see Fig.1).

More specifically, to tackle the first issue, we propose a more label-friendly single-frame supervision (one-click for each abnormal event) [38 , 20 , 67 , 8 , 21] in the domain of video anomaly detection instead of the prohibitive frame-by-frame annotation. Following this labeling paradigm, we manually make single-frame annotations for the exsiting two largest VAD datasets, e . g., UCF-Crime [43] and XD-Violence [55]. To address the second problem of lacking explainability, we construct a large amount of anomaly-awared instruction conversation data for the finetuning of Multimodal LLM. We leverage the single-frame annotated videos and exsiting off-the-shell large foundation model to build an efficient semi-automated data engine. This data engine can be divided into three main steps: 1) Data Collection: gathering video data, primarily from open-source datasets. 2) Annotation Enhancement: generate reliable video event clips around the single-frame annotated frames and give textual descriptions to them through human effort or foundation models. 3) Instruction Construction: utilizing powerful LLM with open-world knowledge to generate explanable analysis in the context of the enhanced video annotations. Subsequently, the obtained analysis is filted manually and structured into conversational format. After on the above steps, a new benchmark containing single-frame temporal annotations and explanatory text descriptions is constructed, and we name the final obtained dataset as VAD-Intruct50k .

Built upon the proposed VAD-Intruct50k, we develop a customized solution for interpretable video anomaly detection, which has three key components, i . e ., Video Encoder, Temporal Sampler and Multi-modal LLM. The Video Encoder and Multi-modal LLM are used to encode the input video and generate text response to the input text prompt, respectively. Additionally, the Temporal Sampler is used to predict the abnormal scores of video frames and sample high-responsive parts as the input for Multi-modal LLM, which is lightweight and enables effient inference. Specially, these three

components can be replaced by any other Video-MLLMs or VAD-Networks. Our primary focus is on how to construct a supervised multi-modal dataset to train these components. Extensive experiments demonstrate that our Holmes-VAD achieve outstanding performance in video anomaly detection and can provide detailed explanations for the detected abnormal events.

To summarize, our major contributions are as follows:

- We propose Holmes-VAD, a video anomaly detection system that is capable of identifying anomalies and providing insightful explainations across even hour-long videos.
- To bridge the dataset gap toward an unbiased and explanable VAD system, we introduce VAD-Intruct50k, a large-scale multimodal video anomaly detection datasets, including single-frame annotations for untrimmed videos, and a large amount of instruction conversation data for trimmed abnormal/normal video clips.
- Extensive quantitative and qualitative experiments demonstrate that the proposed HolmesVAD achieves superior performance and interpretability over recent state-of-the-art methods.

## 2 Related Works

Video Anomaly Detection. This task aims to temporally detect abnormal frames in a long untrimmed video [2 , 39 , 24 , 33 , 49 , 14]. The early VAD attempts are based on hand-crafted features [2 , 18 , 70 , 39 , 33 , 24]. Recently, deep learning approaches [14 , 60 , 37] have become dominant in Video Anomaly Detection (VAD), broadly classified into unsupervised, weakly-supervised, and fully-supervised methods. Unsupervised methods train only on normal videos to learn normal patterns and are often designed as reconstruction-based [14 , 58 , 12 , 60], prediction-based [30], or a combination [31]. Some methods [64 , 44 , 47] also explore a fully unsupervised setting, including both unlabeled normal and abnormal videos in the training set. Weakly-supervised methods [43 , 71 , 10 , 55 , 45 , 23 , 54 , 72 , 37 , 68] use both normal and abnormal videos with video-level annotations. Fully-supervised methods [29 , 19] are less common due to the high cost of precise frame-level annotations.

Multi-modal Large Language Model. The universal and powerful conversational capabilities of ChatGPT [1] have inspired the entire AI community. This has prompted the emergence of the open-source Large Language Models (LMMs), such as LLaMA [46], Vicuna [7], and Mistral [16], based on autoregressive models [48], they are pretrained and instruction tuned via large amounts of text tokens, thus posses universal and powerful text generation capabilities. Recently, Multi-modal LLMs [59 , 26 , 74 , 28 , 27 , 69 , 62 , 9 , 4 , 51] empower LLMs with visual understanding capabilities. Additionally, MLLMs for videos (e.g., VideoChat [22], Video-LLaMA [66], and Video-LLaVA [25]) pave the way for multi-modal temporal understanding.

Multi-modal Video Anomaly Detection. Large-scale visual-language pretrained models such as CLIP [42] serve as a bridge between visual and textual modalities. Some recent works [41 , 17 , 57 , 61] in the realm of video anomaly detection have leveraged textual information as prompts to enhance the model's anomaly representation. Based on this, [56] firstly proposed the open vocabulary VAD task. Furthermore, [65] extracted captions from video frames using a caption model and designed prompts for LLMs to provide anomaly scores. However, these approaches primarily focus on generating anomaly scores and lack fine-tuning on large-scale domain-specific instruction datasets, resulting in their performance being highly dependent on the base LLMs.

## 3 VAD-Instruct50k Benchmark

In this section, we will illustrate the process of VAD-Instruct50k dataset generation. Firstly, the data collection process of VAD-Instruct50k will be presented. Subsequently, we will elaborate on how to enhance the annotations of the collected videos. Finally, the generation process of the instruction conversation data will be introduced. The overall pipeline of the data engine is shown in Fig. 2 .

## 3.1 Data Collection

We first collect videos from the training sets of the two largest weakly-supervised VAD datasets, UCF-Crime [43] and XD-Violence [55], because their video quantity far exceeds that of other existing datasets [50 , 32 , 34], and their video-level annotations provide a solid foundation for further data

Figure 2: Data engine for the proposed VAD-Instruct50k. We collect numerous abnormal/normal videos from exsiting datasets, following by a series of annotation enhancement including temporal single-frame annotation, event clips generation and event clips captioning. Then we construct the instruction data by prompting the powerful LLM with the enhanced annotation. Throughout the pipeline, manual work and large fundation models coordinated with each other to ensure efficiency and quality in construction.

![Image](artifacts/image_000001_d07d3e7dd80583a460815245e735adb2b5c2585a605ef3de35139df6aea68136.png)

processing. After filtering out some low-quality videos via human inspection, we collected a total of 5547 untrimmed videos, include 810/800 abnormal/normal videos from UCF-Crime and 1905/2032 abnormal/normal videos from XD-Violence.

## 3.2 Annotation Enhancement

The collected videos from UCF-Crime [43] and XD-Violence [55] only offers video-level anomaly labels, which denotes whether the video includes anomalies. Going beyond these coarse annotations, we purify these annotations to enable more discriminative anomaly detection model training.

Temporal single-frame annotation. We adopt an efficient temporal annotation method involving sparse single-frame annotation for the collected abnormal videos, inspired by [40 , 38 , 67 , 20 , 21 , 68] that use this approach to balance model performance and annotation cost. Specifically, we annotate only one frame for each abnormal event in the video 1 . Through this process, we collect an average of 2.35 single-frame annotations per video.

Event clip generation. Based on the single-frame annotation, we design a reliable pseudo frame-level label generation method and leverage it to train a VAD network ϕ s 2 . For each abnormal video with single-frame annotations G = {gi} N g and its anomaly score estimated by the trained VAD network, we generate multiple anomaly event proposals around the annotated frame. Formally, each proposal is represented via a starting and ending timestamp, i.e. , s and e. For each normal video, we randomly extract several normal event proposals. After this process, we collect all trimmed event clips with anomaly labels: E = {si, ei, yi} N e , where yiis set to the anomaly class of the video (e . g ., Explosion) if the event clip is from an abnormal video, otherwise, it is set to Normal .

Event clip captioning. To fully extract semantic information from the event clips, we utilize a video-based multimodal large language model (MLLM) [25] to generate detailed captions for each event clip. We also include the SurveillanceVision dataset [63], which provides manually annotated detailed fine-grained event descriptions for video clips from UCF-Crime [43]. After combining these resources, we obtain all event clips with corresponding captions c and anomaly labels: E = {si, ei, yi, ci} N e i .

1 More details about the annotation process are illustrated in Sec. A.2 of the Appendix.

2 See Sec. A.3 of the Appendix for more details about the network.

Figure 3: Overview of Holmes-VAD . Holmes-VAD takes untrimmed video and user prompt as inputs, and takes the anomaly scores and explanation for detected anomalies outputs. The Temporal Sampler takes class tokens of frames as input and estimates the anomaly scores, and the dense visual tokens are resampled accroding to their anomaly scores before entering the projector.

![Image](artifacts/image_000002_ef6cfaafef4ad39edeb13717b73c3ef105a95481515b17093bb55c610408d807.png)

## 3.3 Instruction-tuning Data Construction

The process of annotation enhancement effectively fills the gap of insufficient information in the original video-level annotation. However, there is still a lack of anomaly-awared explanation for these event clips, i . e ., what is the anomaly and why. To address this issue, we utilize the powerful LLM with sufficient open-world knowledge for further instruction dataset construction. Technically, for each event clip in E, we design a task prompts Pt combined with the referenceable anomaly context, i . e ., the abnormal label yi and the detail caption ci. Then we input the combined prompt into the LLM M to make a judgment on anomalies in the video clip and providing an explanation. The generated response is paired with a corresponding anomaly-awared quesion Pd, result in an instruction item:

<!-- formula-not-decoded -->

We use Llama3-Instruct-70B [3] as M here because of its open-source availability and comparable performance to GPT4. We design multiple Pd to ensure the diversity of the instruction data, a typical prompts of Pd is: "&lt;video&gt;\n Are there any unexpected or unusual events in the video clip?".

## 4 Holmes-VAD

Utilizing the proposed VAD-Intruct50k dataset for training, we develop a customised solution for interpretable video anomaly detection, namely Holmes-VAD, which has three key components, Video Encoder, Temporal Sampler and Multi-modal LLM with tunable LoRA [15] modules (see Fig. 3).

## 4.1 Model Architecture

Visual Encoder. We utilize the frozen video encoder in LanguageBind [73] following [25]. It inherits the ViT-L/14 structure from CLIP [42], we refer to it as ϕ v . Different from the orginal ViT, it models the temporal relationship between frames through additional self-attention layer in the temporal dimension. Give a video frame sequence V ∈ R N×H×W×C , the output features of each frame can be denotes as follow:

<!-- formula-not-decoded -->

where f
i cls f
i indicates the class token feature of i-th video frame, f
i j f
i (j ∈ {1 , 2, ..., Np Np }) denotes the visual embedding of each patch, and Np Np reperesents the number of patches of each frame.

Temporal Sampler. Due to the excessive computational burden caused by numerous visual tokens in video, past video-based MLLM approaches [22 , 66 , 25] have resorted to uniform temporal frame sampling of videos, e . g., 8 frames. This method is clearly unsuitable for long videos in video anomaly detection task, as it increases the probability of ignoring key information. [65] conduct dense anomaly detection via MLLM in a frame-by-frame mode, which also inevitably leads to a large amount of redundant computation. To address this issue, we first input the dense video frames into the visual encoder, then we introduce the trained VAD network in 3.2 here, which receives the cls token of the video frames f cls 1 , f 
2 cls f 
2 , ..., f 
N cls f 
N and outputs anomaly scores s1, s2, ..., sN :

<!-- formula-not-decoded -->

where ϕ s denotes the trained VAD network.

Then, we sample the video tokens according to the anomaly scores. Specifically, only the tokens fk from frames with corresponding anomaly score sk above a set threshold θ are then fed into the subsequent network:

<!-- formula-not-decoded -->

where F s denotes the sampled sparse visual tokens from the original dense visual tokens F d . In this way, the model can generate anomaly-awared response to long untrimmed video.

Projector and LLM. To enable the LLM to understand the features output by the visual encoder, a projector ϕproj composed of two layers of MLPs is designed between them, after this, the feature dimention is aligned with the input dimension of LLM. We utilize Vicuna [7] as our LLM following [25].

<!-- formula-not-decoded -->

where T0 T0:i represents the input text tokens to LLM and Ti+1 indicates the predicted next token. ϕproj and ϕT represents the Projector and the Text Encoder, respectively. [· , · ] denotes concatenation.

## 4.2 Training

Training of the Temporal Sampler. In this stage, we only train the Temporal Sampler under the single-frame supervision. In essence, we employed a pseudo-labeling supervision strategy. The pseudo-labels are initialized through single-frame annotations during the training process and are online updated around the annotated frames 3 . We use the generated pseudo label to supervise the predicted anomaly score, which can effectively reduce the bias of the temporal sampler towards easily confued normality.

Instruction Tuning. During this stage, we take the trimmed event clips as input and do not perform Temporal Sampler because each clip has been labeled as abnormal or normal. In this stage, we train the projector and use LoRA [15] to fine-tune the Multi-modal LLM. We conduct different tuning strategy and compare them in the next section. Given the projected visual features Fv Fv and the textual input embedding Ft, the LLM decode them into a sequence words A. we follow mainstream works to use the original auto-regressive training objective. The objective aims to maximize the likelihood of generating the ground truth answer sequence given the input features, encouraging the model to produce coherent and accurate responses based on the input features.

## 5 Experiments

In this section, we conduct extensive experiments to thoroughly demonstrate the capabilities of our proposed model, i.e. , Holmes-VAD .

## 5.1 Experiment Setup

Datasets. We conduct the comparative experiments on two standard VAD datasets, namely, UCFCrime [43] and XD-Violence [55]. (1) UCF-Crime [43] comprises 1900 untrimmed videos totaling 128 hours from outdoor and indoor surveillance cameras. It encompasses 13 classes of real-world anomalies, including Abuse , Explosion , Fighting, and Shooting. In the weakly-supervised setting,

3 More details about the generation of pseudo labels can be found in the Appendix.

Table 1: Comparision with state-of-the-art Video Anomaly Detection approches. We include semisupervised (Semi.) methods, unsupervised (Un.) methods, weakly-supervised (W.) methods and some other methods. "∗" represents the result reported in [65].

| Methods                     | Backbone                    | Supervision                 | Explanation                 | XD-Violence                 | UCF-Crime                   |
|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
|                             |                             | p                           | p                           | AP/%                        | AUC/%                       |
| Non-explainable VAD         | Non-explainable VAD         | Non-explainable VAD         | Non-explainable VAD         | Non-explainable VAD         | Non-explainable VAD         |
| Conv-AE [14]                | -                           | Semi.                       | ✗                           | 27.25                       | 50.60                       |
| GODS [49]                   | I3D                         | Semi.                       | ✗                           | N/A                         | 70.46                       |
| GCL [64]                    | ResNext                     | Un.                         | ✗                           | N/A                         | 71.04                       |
| DYANNET [44]                | I3D                         | Un.                         | ✗                           | N/A                         | 84.50                       |
| MIST [10]                   | I3D                         | W.                          | ✗                           | N/A                         | 82.30                       |
| Wu et al. [55]              | I3D                         | W                           | ✗                           | 78.64                       | 82.44                       |
| RTFM [45]                   | I3D                         | W                           | ✗                           | 77.81                       | 84.30                       |
| MSL [23]                    | I3D                         | W                           | ✗                           | 78.28                       | 85.30                       |
| S3R [54]                    | I3D                         | W.                          | ✗                           | 80.26                       | 85.99                       |
| MGFN [6]                    | I3D                         | W.                          | ✗                           | 79.19                       | 86.98                       |
| UR-DMU [72]                 | I3D                         | W.                          | ✗                           | 81.66                       | 86.97                       |
| CLIP-TSA [17]               | ViT                         | W.                          | ✗                           | 82.19                       | 87.58                       |
| VadCLIP [57]                | ViT                         | W.                          | ✗                           | 84.51                       | 88.02                       |
| Yang et al. [61]            | ViT                         | W.                          | ✗                           | 83.68                       | 87.79                       |
| Wu et al. [56]              | ViT                         | Open-Vocabulary             | ✗                           | 66.53                       | 86.40                       |
| Explainable Multi-modal VAD | Explainable Multi-modal VAD | Explainable Multi-modal VAD | Explainable Multi-modal VAD | Explainable Multi-modal VAD | Explainable Multi-modal VAD |
| ZS CLIP [42]
 ∗             | ViT                         | Training-Free               | ✓                           | 17.83                       | 53.16                       |
| ZS IMAGEBIND [11]
 ∗        | ViT                         | g
 Training-Free            | ✓                           | 25.36                       | 55.78                       |
| LLAVA-1.5 [27]
 ∗           | ViT                         | g
 Training-Free            | ✓                           | 50.26                       | 72.84                       |
| LAVAD [65]                  | ViT                         | g
 Training-Free            | ✓                           | 62.01                       | 80.28                       |
| Holmes-VAD (Ours)           | ViT                         | Instruction-Tuned           | ✓                           | 90.67                       | 89.51                       |

there are 1610/290 videos for training/testing, with the training set consisting of 810 abnormal videos and 800 normal videos, respectively. (2) XD-Violence [55] is the largest VAD benchmark, comprising 4754 videos totaling 217 hours sourced from surveillance, movies, car cameras, and games. It encompasses 6 anomaly classes: Abuse , Car Accidents , Explosions , Fighting , Riots, and Shooting. The training/testing video count stands at 3954/800, adhering to a weakly-supervised framework. The training set comprises 1905 abnormal videos and 2049 normal videos, respectively.

Metrics. To evaluate the anomaly detection performance of the temporal sampler, we use the Area Under the Curve (AUC) as the main evaluation metric for UCF-Crime following [45 , 54 , 23 , 72 , 68]. Meanwhile, the AUC of the frame-level precision-recall curve (AP) is utilized for XD-Violence. To evaluate the quality of explanation response, we randomly extract 86 abnormal/normal video segments from the test videos of UCF-Crime and XD-Violence, and then invite 10 users to vote on the responses of different models from 3 aspects include Judgement Accuracy (JA), Content Perception (CP) and Anomaly Explanatory (AE). Please see the Appendix for more details about the metrics.

Implementation details. In our study, we take the ViT in LanguageBind model [73] as the Video Encoder and initialize the Multi-modal LLM with Video-LLaVA [25]. UR-DMU [72] serves as the foundation structure for our Temporal Sampler. To optimize the Temporal Sampler, we ramdomly sample one frame at 16-frame intervals, and Adam optimizer with a learning rate of 1e-4 is adopted. Note that when evaluating performance on XD-Violence and UCF-Crime, only videos in the corresponding training sets are used to train our model for fair comparisons. For instruction tuning, we train with a batch size of 128 for 1 epoch, using the AdamW optimizer with cosine learning rate decay and a warm-up period, setting the projector's learning rate to 2e-5. The LoRA [15] parameters are set as: r=64, α=128, and learning rate=2e-4. The abnormal threshold θ is set to 0.8 during inference. Experiments are conducted on 2 NVIDIA A100 GPUs.

## 5.2 Main Results

We compare our method with state-of-the-art methods, including semi-supervised methods [14 , 49], unsupervised methods [64 , 44], weakly-supervised methods [45 , 23 , 54 , 72 , 17 , 57] and recently training-free method [65]. We have indicated their backbones, supervision methods, and performance on the UCF-Crime and XD-Violence datasets, as shown in Table 1. Our method has an AP of 90.67% on XD-Violence and an AUC of 89.51% on UCF-Crime, significantly outperforming the

prior state-of-the-art methods, which demonstrates that our method can generate less biased anomaly scores. It is worth noting that while achieving precise localization of anomalies, Holmes-VAD is also capable of providing explanations and analysis for the detected anomalies by the model, a feature unavailable in existing non-explainable VAD methods. Although LAVAD [65] has explainability, the training-free large language model lacks an understanding of anomaly knowledge due to the limitation of insufficient supervised data.

## 5.3 Analytic Results

Table 2: Human evaluation on models under different training settings.

| Training Strategy        |   Average Words of Model Reponse  |   JA(%)  |   CP(%)  |   AE(%) |
|--------------------------|-----------------------------------|----------|----------|---------|
| Training-free            |                             38.29 |     65.1 |     11.6 |    15.9 |
| Projector                |                             40.84 |     81.4 |     27.2 |    32.2 |
| Projector+LoRA (Default) |                             46.13 |     86   |     61.2 |    51.9 |

Table 3: Ablation study of backbone and supervision in Temporal Sampler. Table 4: Effect of random temporal shifting of single-frame annotations.

| Backbone    | Single-frame    |   XD-AP(%)  |   UCF-AUC(%) |   Shifted Timestamp  |   XD-AP(%)  |   UCF-AUC(%) |
|-------------|-----------------|-------------|--------------|----------------------|-------------|--------------|
| I3D [5]     | ✗               |       82.4  |        86.54 |                   0  |       90.67 |        89.51 |
| ViT [73]    | ✗               |       84.96 |        84.61 |                  10  |       90.55 |        89.45 |
| I3D [5]     | ✓               |       89.4  |        90.8  |                  50  |       90.45 |        89.32 |
| ViT [73]    | ✓               |       90.67 |        89.51 |                 100  |       90.12 |        88.95 |

Table 5: Temporal Sampler v.s. Uniform Sampler. We averaged the inference time of all test videos.

| Sampling Strategy          |   XD-AP(%)  |   UCF-AUC(%)  |   Avg. Infer Speed (second per video) |
|----------------------------|-------------|---------------|---------------------------------------|
| Uniform Sampler            |       67.25 |         78.38 |                                 32.82 |
| Temporal Sampler (Default) |       90.67 |         89.52 |                                  4.24 |

Influence of varied training strategies on anomaly explanation. We conduct a user study to evaluate three different training strategies over 86 test samples and 10 volunteers: a) Trainingfree: no fine-tuning; b) Projector: fine-tuning on VAD-Instruct50k, only training the projector while keeping the Multi-modal LLM fixed; c) Projector+LoRA: fine-tuning on VAD-Instruct50k, training the projector and using LoRA [15] to fine-tune the Multi-modal LLM. As shown in Table 2 , Projector+LoRA provide the most detailed response (46.13 words in average) and reaches the highest Judgement Accuracy (86.0%). Addtionally, it also achieves the highest voting rate, including 61.2% on Content Perception and 51.9% on Anomaly Explanatory, these demonstrate better interpretability by fine-tuning Multi-modal LLM on VAD-Instruct50k.

Backbone and supervision matters in Temporal Sampler. In Table 3, we ablate the impact of video backbone and the supervision for Temporal Sampler. We use UR-DMU [72] as our baseline method. The results indicate that on XD-Violence dataset, LanguageBind [73] as a backbone outperforms I3D [5] significantly, whereas the opposite is observed on UCF-Crime. Additionally, single-frame supervision significantly enhances performance regardless of the backbone used, demonstrating the effectiveness of point supervision in improving anomaly localization capabilities.

Influence of perturbed single-frame annotations. To assess the robustness of our method to the perturbed temporal position of single-frame annotation, we introduce varied temporal timestamp shifts to the original positions of the annotated frames. As shown in Table 4, there is no significant performance degradation of our model under perturbed annotation positions, indicating that our method possesses a notable tolerance towards variations in degraded supervision.

Temporal Sampler v.s. Uniform Sampler. We replace the Temporal Sampler with Uniform Sampler while maintaining the frame rate. The video is then divided into non-overlapping clips, which are sequentially fed into the Multimodal LLM to output results. If the output is "Yes" the anomaly scores of all frames in the input segment are set to 1, otherwise, they are set to 0. Finally, we compare the

Figure 4: Qualitative results. We compare our interpretability results with Video-LLaVA [36] (without instruction tuning). Correct and wrong explanations are in green and red, respectively.

![Image](artifacts/image_000003_1fdc994f9be7c4ceac961b37547c6cc9976d8e68dc24d093592b266d72328229.png)

detection performance and inference efficiency in Table 5. The results demonstrate that the Temporal Sampler ensures higher inference efficiency while maintaining accurate detection results.

Qualitative comparision. To provide a more intuitive understanding of the capabilities of MLLM in explaining complex anomalies, we provide qualitative comparisons between Holmes-VAD and Video-LLaVA in Fig. 4. The results demonstrate that Holmes-VAD can accurately identify anomalies in videos and provide specific explanations for conflicts in sports competitions, explosions, and accidents captured by car cameras (Abnormal Cases). Even for normal videos, Holmes-VAD exhibits robust analytical abilities, correcting erroneous responses from the Temporal Sampler (Normal Cases). These findings highlight the effectiveness and advantage of Holmes-VAD in perceiving video events and analyzing anomalies.

## 6 Conclusion

In this paper, we introduce a video anomaly detection system called Holmes-VAD to address the biases and lack of interpretability in existing anomaly detection methods. By introducing a more efficient labeling paradigm and constructing a large-scale multimodal video anomaly detection dataset, VAD-Instruct50k, we validated the generality and interpretability of Holmes-VAD. Through extensive experiments, we positioned Holmes-VAD as a valuable tool for real-world applications.

Limitation and Future work. Despite the human effort for filtering the noise instruction data during constructing the VAD-Instruct50k dataset, the reliance on off-the-shelf video captioning models for generating video description may not always capture the nuances and context-specific information. This is a trade-off we have made between labeling costs and efficiency, we believe that the quality of data is no less important than the quantity of data, and we plan to further enhance data quality and quantity within acceptable labor costs in the future. Furthermore, although we control the length of the video input to the Multi-modal LLM through Temporal Sampler and accurately analyze abnormal content in the trimmed video clips, there is still a lack of an effective solution for Multimodal LLM to understand long-term video anomalies without compromising its image-level perceptual capabilities. We leave these for our future exploration.

Acknowledgement This work is supported by the National Natural Science Foundation of China under grant U22B2053 and 623B2039.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Amit Adam, Ehud Rivlin, Ilan Shimshoni, and Daviv Reinitz. Robust real-time unusual event detection using multiple fixed-location monitors. IEEE transactions on pattern analysis and machine intelligence , 30(3):555–560, 2008.
- [3] AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/blob/ main/MODEL\_CARD.md .
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023.
- [5] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, pages 6299–6308, 2017.
- [6] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 387–395, 2023.
- [7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.
- [8] Ran Cui, Tianwen Qian, Pai Peng, Elena Daskalaki, Jingjing Chen, Xiaowei Guo, Huyang Sun, and Yu-Gang Jiang. Video moment retrieval from text queries via single frame annotation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 1033–1043, 2022.
- [9] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024.
- [10] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. Mist: Multiple instance self-training framework for video anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14009–14018, 2021.
- [11] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023.
- [12] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1705–1714, 2019.
- [13] Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, and Jinqiao Wang. Anomalygpt: Detecting industrial anomalies using large vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1932–1940, 2024.

- [14] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 733–742, 2016.
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 , 2021.
- [16] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [17] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. Clip-tsa: Clip-assisted temporal selfattention for weakly-supervised video anomaly detection. In 2023 IEEE International Conference on Image Processing (ICIP), pages 3230–3234. IEEE, 2023.
- [18] Jaechul Kim and Kristen Grauman. Observe locally, infer globally: a space-time mrf for detecting abnormal activities with incremental updates. In 2009 IEEE conference on computer vision and pattern recognition , pages 2921–2928. IEEE, 2009.
- [19] Federico Landi, Cees GM Snoek, and Rita Cucchiara. Anomaly locality in video surveillance. arXiv preprint arXiv:1901.10364, 2019.
- [20] Pilhyeon Lee and Hyeran Byun. Learning action completeness from points for weakly-supervised temporal action localization. In ICCV, pages 13648–13657, 2021.
- [21] Hanjun Li, Xiujun Shu, Sunan He, Ruizhi Qiao, Wei Wen, Taian Guo, Bei Gan, and Xing Sun. D3g: Exploring gaussian prior for temporal sentence grounding with glance annotation. arXiv preprint arXiv:2308.04197, 2023.
- [22] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [23] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 1395–1403, 2022.
- [24] Weixin Li, Vijay Mahadevan, and Nuno Vasconcelos. Anomaly detection and localization in crowded scenes. IEEE transactions on pattern analysis and machine intelligence, 36(1):18–32, 2013.
- [25] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [26] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023.
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [29] Kun Liu and Huadong Ma. Exploring background-bias for anomaly detection in surveillance videos. In Proceedings of the 27th ACM International Conference on Multimedia, pages 1490–1499, 2019.
- [30] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6536–6545, 2018.
- [31] Zhian Liu, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13588–13597, 2021.
- [32] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In Proceedings of the IEEE international conference on computer vision, pages 2720–2727, 2013.
- [33] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In Proceedings of the IEEE international conference on computer vision, pages 2720–2727, 2013.

- [34] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In Proceedings of the IEEE international conference on computer vision, pages 341–349, 2017.
- [35] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In Proceedings of the IEEE international conference on computer vision, pages 341–349, 2017.
- [36] Hui Lv and Qianru Sun. Video anomaly detection and explanation via large language models. arXiv preprint arXiv:2401.05702, 2024.
- [37] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8022–8031, 2023.
- [38] Fan Ma, Linchao Zhu, Yi Yang, Shengxin Zha, Gourab Kundu, Matt Feiszli, and Zheng Shou. Sf-net: Single-frame supervision for temporal action localization. In ECCV, pages 420–437. Springer, 2020.
- [39] Ramin Mehran, Alexis Oyama, and Mubarak Shah. Abnormal crowd behavior detection using social force model. In 2009 IEEE conference on computer vision and pattern recognition, pages 935–942. IEEE, 2009.
- [40] Pascal Mettes, Jan C Van Gemert, and Cees GM Snoek. Spot on: Action localization from pointlysupervised proposals. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14, pages 437–453. Springer, 2016.
- [41] Yujiang Pu, Xiaoyu Wu, and Shengjin Wang. Learning prompt-enhanced context features for weaklysupervised video anomaly detection. arXiv preprint arXiv:2306.14451, 2023.
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [43] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6479–6488, 2018.
- [44] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. Dyannet: A scene dynamicity guided self-trained video anomaly detection network. In Proceedings of the IEEE/CVF Winter conference on applications of computer vision, pages 5541–5550, 2023.
- [45] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4975–4986, 2021.
- [46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [47] Anil Osman Tur, Nicola Dall'Asen, Cigdem Beyan, and Elisa Ricci. Exploring diffusion models for unsupervised video anomaly detection. In 2023 IEEE International Conference on Image Processing (ICIP), pages 2540–2544. IEEE, 2023.
- [48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [49] Jue Wang and Anoop Cherian. Gods: Generalized one-class discriminative subspaces for anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8201–8211, 2019.
- [50] Shu Wang and Zhenjiang Miao. Anomaly detection in crowd scene. In IEEE 10th International Conference on Signal Processing Proceedings, pages 1220–1223. IEEE, 2010.
- [51] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.
- [52] Xiang Wang, Shiwei Zhang, Jun Cen, Changxin Gao, Yingya Zhang, Deli Zhao, and Nong Sang. Clipguided prototype modulating for few-shot action recognition. International Journal of Computer Vision , pages 1–14, 2023.

- [53] Xiang Wang, Shiwei Zhang, Hangjie Yuan, Yingya Zhang, Changxin Gao, Deli Zhao, and Nong Sang. Few-shot action recognition with captioning foundation models. arXiv preprint arXiv:2310.10125, 2023.
- [54] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Self-supervised sparse representation for video anomaly detection. In European Conference on Computer Vision, pages 729–745. Springer, 2022.
- [55] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 322–339. Springer, 2020.
- [56] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Openvocabulary video anomaly detection. arXiv preprint arXiv:2311.07042, 2023.
- [57] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. arXiv preprint arXiv:2308.11681, 2023.
- [58] Dan Xu, Yan Yan, Elisa Ricci, and Nicu Sebe. Detecting anomalous events in videos by learning deep representations of appearance and motion. Computer Vision and Image Understanding, 156:117–127, 2017.
- [59] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.
- [60] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on keyframes for video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14592–14601, 2023.
- [61] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. arXiv preprint arXiv:2404.08531, 2024.
- [62] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [63] Tongtong Yuan, Xuange Zhang, Kun Liu, Bo Liu, Chen Chen, Jian Jin, and Zhenzhen Jiao. Towards surveillance video-and-language understanding: New dataset, baselines, and challenges, 2023.
- [64] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. Generative cooperative learning for unsupervised video anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14744–14754, 2022.
- [65] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. arXiv preprint arXiv:2404.01014, 2024.
- [66] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.
- [67] Huaxin Zhang, Xiang Wang, Xiaohao Xu, Zhiwu Qing, Changxin Gao, and Nong Sang. Hr-pro: Point-supervised temporal action localization via hierarchical reliability propagation. arXiv preprint arXiv:2308.12608, 2023.
- [68] Huaxin Zhang, Xiang Wang, Xiaohao Xu, Xiaonan Huang, Chuchu Han, Yuehuan Wang, Changxin Gao, Shanjun Zhang, and Nong Sang. Glancevad: Exploring glance supervision for label-efficient video anomaly detection. arXiv preprint arXiv:2403.06154, 2024.
- [69] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.
- [70] Bin Zhao, Li Fei-Fei, and Eric P Xing. Online detection of unusual events in videos via dynamic sparse coding. In CVPR 2011, pages 3313–3320. IEEE, 2011.
- [71] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1237–1246, 2019.

- [72] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. arXiv preprint arXiv:2302.05160, 2023.
- [73] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by languagebased semantic alignment. arXiv preprint arXiv:2310.01852, 2023.
- [74] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592 , 2023.

## A Appendix

## A.1 Broader Impact

The paper proposes a video anomaly detection framework, namely Holmes-VAD, that is capable of temporally identifying anomalies accurately and providing insightful explainations across even hour-long videos. Additionally, this paper provides VAD-Intruct50k, a large scale multimodal video anomaly detection datasets, including single-frame annotations for untrimmed videos, and a large amount of instruction conversation data for trimmed abnormal/normal video clips.

The positive societal impacts of the work include:

- Improved public safety: The development of more accurate and interpretable video anomaly detection systems can enhance public safety by enabling quicker and more precise identification of anomalies in surveillance videos, such as criminal activities or accidents.
- Advancement in supervised and open-world VAD research: The proposed VADIntruct50k dataset provide a). accurate temporal timestamp of the abnormal events in videos, and b). video-explanation pair for both abnormal and normal video clips, which can pave the way for further supervised and open-world research in the video anomaly detection area.

The negative societal impacts may include:

- Privacy concerns: The use of video surveillance technology, especially in public spaces, raises concerns about privacy and the potential for intrusive monitoring of individuals without their consent.
- Disregard for minor anomalies: Despite efforts to reduce bias in anomaly detection, there is still a risk of disregard for subtle anomalies such as stealing in the supermarket, leading to potential undetected anomalies.

Consequently, researchers should adhere to relevent laws and regulations, and strive to avoid using our model or dataset for any improper invasion of privacy. Meanwhile, all our model and data will be only used for research purpose to avoid the potential negative societal impacts.

## A.2 Process of single-frame annotation.

Figure 5: Screenshot of the single-frame annotation interface.

![Image](artifacts/image_000004_42d1c8fe4d8b8b730b7a87976c66a6dd2eec77613feca98217bbc3a97fdcd7f6.png)

Annotation tool. We develop an interface designed specifically for single-frame annotation in videos, as shown in Fig. 5. This interface makes it easier to navigate through video lists, adjust video progress rapidly, and automatically record timestamps for annotating individual frames. Furthermore, it enables the preview of annotated frames. By clicking on the annotated frame ID, the video progress automatically synchronizes with the corresponding temporal position. These features greatly

Figure 6: Examples of single-frame annotation.

![Image](artifacts/image_000005_e1cbd2f2b7753731d7e818e81534040c807b81c73afa6ce8aac66d6bcb6405c8.png)

streamline the annotation process, enhancing convenience and efficiency. If annotators come across any errors or need to make adjustments, they can delete incorrect annotations and proceed with re-annotation.

Quality control. We initially divide the entire dataset into various portions and distribute them among different annotators for labeling. Once the first round of annotations is completed, we proceed with a secondary review of the video annotations to eliminate incorrect or redundant annotations. In addition, we include ignored clicks to minimize the possibility of overlooking potential anomalies. This process ensures the Reliability and Comprehensiveness of the single-frame annotations.

Examples of single-frame annotation. To facilitate a better understanding of the annotation process, we offer several examples of annotated videos in Fig. 6 .

## A.3 Model architecture and training details of the Temporal Sampler.

Model architecture. We use UR-DMU [72] as the VAD network in our Temporal Sampler. As shown in Fig. 7, UR-DMU utilizes a Global and Local Multi-Head Self Attention (GLMHSA) module to capture both long-range and short-range temporal relationships among video snippets. Furthermore, UR-DMU introduces two memory banks to store and differentiate abnormal and normal prototypes, thereby maximizing the margins between these two representations. In order to learn discriminative representations, UR-DMU employs triplet loss to increase the feature distance after interacting with different memories. Simultaneously, it utilizes KL loss to constrain the normal memory to follow a Gaussian distribution, accounting for the variance introduced by noise. Thus, the base loss function for the UR-DMU baseline is defined as follows:

Figure 7: Architecture of the Temporal Sampler.

![Image](artifacts/image_000006_b14b1464e1a0e37d26e93d3d0bcc20b2673f146737bf4f8241835ec13febe12f.png)

<!-- formula-not-decoded -->

Training details. During the training stage of the Temporal Sampler, we leverage the sparse singleframe annotations to generate reliable dense snippet-level pseudo label. As illustrated in Alg. 1 , we employ a dynamic threshold and perform local bidirectional mining based on the single-frame annotations. Snippets with anomaly scores exceeding a specific proportion of the annotated snippet's score are identified as pseudo anomaly snippets. We set α = 0 . 9 in our implementation. After mining the pseudo anomaly snippets, we adopt Gaussian function to smooth the binary pseudo label:

<!-- formula-not-decoded -->

## Algorithm 1 Pseudo Label Mining.

```
Input: Anomaly score S ∈ R T , single-frame annotations G = {gi} N g , anomaly ratio α . Output: Pseudo anomaly snippets T a = {ti} N a i . 1: Let T a ← ∅ . 2: for every gi ∈ G do 3: for t = gi to gi − 1 do 4: if S[t] > α · S[gi] , then T a ← t ∪ T a , else break 5: end if 6: end for 7: for t = gi to gi+1 do 8: if S[t] > α · S[gi] , then T a ← t ∪ T a , else break 9: end if 10: end for 11: end for 12: Return T a
```

where r = 0 . 1 indicates the smoothing ratio. We use the generated dense and smooth pseudo label to supervise the predicted anomaly score:

<!-- formula-not-decoded -->

where S and S ˆ denote the predicted anomaly score and the generated pseudo frame-level label, respectively.

## A.4 Details of human evaluation.

Figure 8: Screenshot of the human evaluation interface.

![Image](artifacts/image_000007_008623005d52bc5b7d805a7883a975512de9ea576668f1b2f81fe6cf9b031e3c.png)

To evaluate the quality of explanation response, we randomly extract 86 abnormal/normal video segments from the test videos of UCF-Crime and XD-Violence, and then invite 10 users to vote on the responses of different models from 3 aspects include Judgement Accuracy (JA), Content Perception (CP) and Anomaly Explanatory (AE).

- Judgement Accuracy (JA): Determine whether the model's judgment on anomalies is correct, we extract predictions by matching "Yes"/"No" in the answers, and compare them with the ground truth labels (abnormal/normal). Finally, we calculate the accuracy of the judgments.
- Content Perception (CP): The accuracy and clarity of the model's descriptions of the content, characters, and events in the video scenes, as well as any potential hallucination issues (descriptions of non-existent objects in the video or responses unrelated to the questions).
- Anomaly Explanatory (AE): The model's ability to analyze and interpret abnormal/normal events in the video.

We provide the screenshot of the human evaluation interface in Fig. 8, to ensure a fair selection, the names of the models are not visible to the users, and choices can only be made from anonymous options. In Fig. 9, we provide several test examples, with the results of the Judgement Accuracy (JA), Content Perception (CP) and Anomaly Explanatory (AE).

## A.5 Dats statistical analysis of VAD-Instruct50k

In Table 6, we conduct a statistical analysis of our proposed VAD-Instruct50k and compare it with representative datasets in the VAD field, which shows the significant volume and excellent diversity of our constructed instruction dataset.

Table 6: Datasets Statistics .

| Dataset                | #Videos    | Annotation Type                  | #Queries    | Avg word   |
|------------------------|------------|----------------------------------|-------------|------------|
| CHUK Avenue            | 37         | None                             | N/A         | N/A        |
| ShanghaiTech           | 437        | None/video label                 | N/A         | N/A        |
| UCF-Crime [43]         | 1,610      | video label                      | N/A         | N/A        |
| XD-Violence [55]       | 4,754      | video label                      | N/A         | N/A        |
| UCA [63]               | 1,854      | segment caption                  | 23,542      | 20.15      |
| VAD-Instruct50k (Ours) | 5,547      | single-frame&segment instruction | 51,567      | 44.83      |

Figure 9: Qualitative comparision in human evaluation. We show the results of Judgement Accuracy (JA), Content Perception (CP) and Anomaly Explanatory (AE) above the answer box of each model.

![Image](artifacts/image_000008_fbf6977cfc36c01eddf12685c5b365a556d20842a302fa565a2b0150c6dcac46.png)