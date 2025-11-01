---
title: Video Anomaly Detection and Explanation via Large Language Models
type: other
categories:
- Semi Supervised
github_link:
description: The paper introduces VAD-LLaMA, a novel framework integrating 
  video-based large language models (VLLMs) for threshold-free, explainable 
  video anomaly detection, featuring a Long-Term Context (LTC) module and a 
  three-phase training process that enhances long-range context modeling and 
  minimizes data annotation costs.
benchmarks:
- ucf-crime
- other
authors:
- Hui Lv
- Qianru Sun
date: '2023-10-01'
---

## Video Anomaly Detection and Explanation via Large Language Models

Hui Lv 1 , Qianru Sun 1*

1 Singapore Management University

1 {huilyu, qianrusun}@smu.edu.sg

Figure 1. Prediction scores from a baseline VAD model, and clip descriptions by using VLLMs, for a car accident video (as shown in the middle of the figure). On the score curve, the red dashed lines denote anomaly thresholds. The bottom shows the answers from Video-LLaMA [24] by feeding it with two pairs of video clips and questions, respectively: {Green: a normal video clip, "Is there any anomaly in the video?"} and {Orange: an abnormal video clip, "Is there a car accident? If so, is it an anomaly?"}

![Image](artifacts/image_000000_fad551b68b09ab84734dde47c65971af22ea9e8194f3af5d471fb56cc100096e.png)

threshold given diverse video content as well as abnormal events. For example, as depicted on the top of Figure 1, using different thresholds on the prediction results (scores) of the VAD model yields different detection outcomes. Secondly, with a carefully selected threshold, anomalies are localized along the timeline based on only scores, and these scores provide little information for users to comprehend the contexts or ascertain the reasons behind the anomalies. In this paper, we are interested in the VAD model not merely to automatically identify anomalies but also to provide comprehensive textual explanations. We incorporate the Videobased Large Language Models (VLLMs) into the framework of VAD, Video-LLaMA [24] in our case, and call the method VAD-LLaMA. In the following, we elaborate on the challenges and our solutions.

Well-trained VLLMs (such as Video-ChatGPT [15],

## Abstract

Video Anomaly Detection (VAD) aims to localize abnormal events on the timeline of long-range surveillance videos. Anomaly-scoring-based methods have been prevailing for years but suffer from the high complexity of thresholding and low explanability of detection results. In this paper, we conduct pioneer research on equipping video-based large language models (VLLMs) in the framework of VAD, making the VAD model free from thresholds and able to explain the reasons for the detected anomalies. We introduce a novel network module Long-Term Context (LTC) to mitigate the incapability of VLLMs in long-range context modeling. We design a three-phase training method to improve the efficiency of fine-tuning VLLMs by substantially minimizing the requirements for VAD data and lowering the costs of annotating instruction-tuning data. Our trained model achieves the top performance on the anomaly videos of the UCF-Crime and TAD benchmarks, with the AUC improvements of +3.86% and +4.96%, respectively. More impressively, our approach can provide textual explanations for detected anomalies. Our code is in the Appendix.

## 1. Introduction

Video Anomaly Detection (VAD) is to identify unexpected events in video sequences. It has practical applications that span a multitude of fields including intelligent manufacturing [6], traffic surveillance [7 , 14] and public security [16 , 18]. Conventional VAD methods [13 , 14 , 18 , 19 , 24 , 28] are designed to predict anomaly scores along the timeline of the video, i.e., one anomaly score for each video frame. A higher score indicates a higher possibility of being abnormal on the frame. These anomaly-score-based designs are simple to implement but remain far away from the ideal agents of VAD which should be both automatic (i.e., free from manually-selected thresholds) and explainable (i.e., being able to explain why an event is abnormal).

Firstly, it is not intuitive how to determine the optimal

* Corresponding author

Videochat [9], and Video-LLaMA [24]) can generate detailed captions for any input video. However, there is a discrepancy between VLLMs' and humans' understanding of anomalies. As illustrated in Figure 1, Video-LLaMA [24] identifies several irrelevant objects in the scene as "anomalies", while overlooking the car accident—the real anomaly humans care about. To solve this issue, we propose a novel Video Anomaly Detector (VADor) equipped with the modules from Video-LLaMA [24], and a new method for co-training VADor and VLLMs without needing a large amount of domain-specific data and labels.

This co-training poses two key challenges. The first challenge is that open-sourced VLLMs lack long-range context modeling ability. They are mostly trained on short videos with simple contexts, but the videos of VAD exhibit high context complexities. For example, WebVid [1], commonly used for fine-tuning VLLMs, features an average video length of only 18 seconds, notably shorter compared to the average length of 240 seconds in the VAD dataset UCF-Crime [18]. In long videos, anomalies really depend on long-range video contexts. For example, identifying a burglary event requires consideration of preceding activities, such as the breaking of windows or doors, even if the event itself only displays moving valuables outside. The second challenge is the lack of VAD data and labels. The widely-used VAD dataset, UCF-Crime [18], is used for weakly-supervised VAD, as it offers only videolevel anomaly annotations, i.e., given a video, it has only a one-hot label indicating normal or abnormal. Therefore, it is not intuitive how to generate text-based instruction data to fine-tune VLLMs. Besides, VAD datasets have a small scale, e.g., UCF-Crime contains 1.9K training videos, significantly smaller than the VLLM training datasets such as WebVid [1] containing 10M videos. The fine-tuning of VLLMs on VAD datasets is thus challenging.

To tackle the first challenge, we introduce a novel LongTerm Context (LTC) module in the VADor. The key idea is to integrate the long-term normal/abnormal contexts into the video representation. First, we split a video into multiple clips, and use the video encoder (VE) of Video-LLaMA to extract the features of each clip. Taking the features as input, VADor can output an anomaly score for each clip. Based on the lowest (highest) K anomaly scores, we pick corresponding clip features and stack them into a normal (abnormal) list. The generation of these two lists is implemented as an online operation for each video: every new clip will be immediately evaluated based on its anomaly score to update the lists (or not). Given the "raw" features of the next clip, we integrate the current lists of normal and abnormal features by cross-attention and weighted-sum operations, see Sec. 3.2, i.e., the way we integrate the long-term contexts of the video into the video representation.

To resolve the second challenge, we propose a three- phase training method. The first phase is to train a baseline VADor, based on which we can easily form a new VAD dataset with each frame "annotated" by an anomaly score. In the second phase, we co-train VADor and the proposed LTC on the above dataset. The primary objective here is to incorporate long-term contextual understanding into the LTC and then use it to enhance the video representation of Video-LLaMA in the final phase. The final phase is to fine-tune Video-LLaMA. Based on the above dataset, we manually compose simple textual templates (showcased in Sec. 3.1) to generate instruction-tuning data, and then use the data to train only the projection layer of Video-LLaMA. We avoid fine-tuning the entire Video-LLaMA due to the limited scale of the VAD dataset. Moreover, to prevent overfitting on VAD videos, we incorporate a diverse training sample set, drawing from both the UCF-Crime and WebVid datasets. The latter has been instrumental in the pre-training of Video-LLaMA. Our method enhances the efficiency of training VAD-LLaMA by substantially minimizing the requirements for VAD data and lowering the costs of creating instruction-tuning data. During testing, VAD-LLaMA is capable of not only identifying anomalies from the input video but also outputting textual explanations of the reasons for being abnormal.

Our contributions are thus three-fold. 1) A new approach called VAD-LLaMA that introduces VLLMs for tackling the task of VAD. 2) A novel LTC module that enhances the long video representation ability of existing VLLMs. 3) A novel three-phase training method for the proposed VADLLaMA, by resolving the issues of lacking VAD data and instruction-tuning data.

## 2. Related Work

Video anomaly detection (VAD) has been a prominent research area with diverse real-life applications. However, it remains a challenging task primarily due to the scarcity of anomalous data and labels. Consequently, researchers often turn to Weakly Supervised Video Anomaly Detection (WSVAD) methods to address the VAD problem. These approaches make use of both normal and abnormal training data, relying on weak annotations provided only at the video-level [18]. Multiple instance learning (MIL) is the mainstream paradigm that uses video-level labels for training snippet-level anomaly detectors [5 , 10 , 18 – 20 , 22 , 25 , 29]. Generally, they embrace the two-stage anomaly detection pipeline, which performs anomaly detection upon preextracted features. In particular, Zhong et al. [27] considered the WSVAD task as supervised learning under noise labels and they designed an alternate training procedure to enhance the discrimination of action classifiers. Lv et al. [14] focused on anomaly localization and proposed a higher-order context model as well as a margin-based MIL loss. Li et al. [10] proposed multiple sequence learning, where consecutive snippets with high anomaly scores are selected in MIL learning. More recently, Lv et al. [13] proposed an unbiased MIL framework for removing the context bias. And they integrated feature representation fine-tuning and anomaly detector learning into an end-to-end training fashion. In this paper, we follow the end-to-end manner to tackle the WSVAD problem and make the first effort to introduce VLLMs into VAD for endowing the VAD model with the ability of anomaly description.

Video-based large language models (VLLMs) have demonstrated remarkable language understanding and reasoning abilities, thanks to the ongoing research efforts in exploring the use of LLMs for processing multi-modal problems [4 , 9]. Bain et al. [1] introduced WebVid, a large-scale dataset of short videos with textual descriptions sourced from stock footage sites. Based on it, Li et al. [9] improved image encoders, enabling large models to understand visual content in videos. Su et al. [17] utilized multi-modal encoders to enable large models to understand six modalities. Zhang et al. [24] trained fundamental models to comprehend both the visual and auditory content in videos. In this work, we focus on the visual modal in videos, since most videos in VAD are collected from road surveillance, which falls short of audio signals. By integrating the designed VADor with the VLLMs, we propose a novel approach VAD-LLaMA, which is able to not only detect the anomalies but also explain the details of the anomalies.

## 3. Method

Our VAD-LLaMA architecture is illustrated in Figure 2 and its training method is shown in Figure 3. It aims to adapt the general video representation knowledge of a pretrained large video-language model Video-LLaMA [24] to tackle VAD tasks. Below, we first elaborate on its network architecture in Sec. 3.1. Then, we delve into the specifics of the three-phase training method in Section 3.2 .

## 3.1. Model Architecture

Overview. As depicted in Figure 2, VAD-LLaMA mainly consists of a new VADor, and two pre-trained modules (VE and LLaMA) from Video-LLaMA [24]. The VADor is built upon the VE and includes a novel LTC module and a simple Anomaly Predictor (AP) g consisting of two fully-connected (fc) layers. Besides, VAD-LLaMA learns an adaptor f between the VADor and the LLaMA to align their feature distributions.

VE and Feature Extraction. Given a video sequence, we first divide it into m segments. For each segment, we randomly sample a video clip (consecutive frames), and feed it into the pre-trained VE to extract clip-level features. We denote x i , i ∈ {1, . . . , m} as the VE feature of the i-th clip. In this work, the adopted VE from Video-LLaMA [24] consists of an image encoder (BLIP-2 [8]) and a Video-

Figure 2. The network architecture of the proposed VAD-LLaMA. It consists of a Video Anomaly Detector (VADor) with the LongTerm Context (LTC) module and a simple Anomaly Predictor (AP), a projection layer (called Adaptor), and the pre-trained Video-LLaMA [24] (composed by a Video Encoder (VE) and a LLaMA). The training of VAD-LLaMA is decomposed into three phases, and the trainable and frozen modules vary among different training phases. Training phases are given in Figure 3 .

![Image](artifacts/image_000001_3e5e94a5cffbb5b1d574b9689430d0c7642c848f2ed66bf3a24ee1dd9bdb4853.png)

Qformer, sharing the same architecture with Query Transformer [8]. The image encoder includes a ViT-G/14 from EVA-CLIP [3] and an image-level Query Transformer. As aforementioned, these VE features lack long-term context information, as the VE was pre-trained mainly with short and normal videos.

Long-Term Context (LTC) Module. The LTC module is proposed to solve the above challenge. Specifically, we collect the clip-level VE features with K lowest (highest) anomaly scores and stack them into a normal (abnormal) list. We denote the normal list as N = {nj} K j=1 , and abnormal list as A = {Aj} K j=1 . These two lists are online updated and every new clip will be immediately evaluated based on its anomaly score to update the lists (or not). In addition, we introduce the cross-attention mechanism in the LTC module for integrating the two lists' information into the VE features. Output features of the LTC module are not only taken as inputs into the AP, but also stacked with the VE features to serve as the visual prompts (input embeddings) of LLaMA. Based on the LTC-enhanced features, we are able to derive a more robust VADor and also provide comprehensive video contexts for LLaMA.

Feature Adaptor. In VAD-LLaMA, the Adaptor f (one fc layer) is added to convert the visual prompts into the same dimension with the inputs of LLaMA and align the visual feature distributions with the pre-trained LLaMA.

LLaMA. In this work, we adopt the LLaMA of version vicuna-7b [26]. It's important to highlight that the finetuning of LLMs is based on the instruction-tuning data [1].

Figure 3. The training phase of VAD-LLMs consists of three phases. 1) VAD baseline training, 2) VAD co-training with LTC, and 3) Instruction-tuning Adaptor. In the LTC module, N and A represent the long-term normal and abnormal feature lists, respectively. The red arrow denotes the generation process from anomaly scores to pseudo instructions with text templates.

![Image](artifacts/image_000002_2c208f7985052d2b15cfd3c3959f19fd6e93671c76c4b49dccbd699895e84f65.png)

Typically, this data is comprised of video instruction pairs, where each pair includes a textual instruction corresponding to the content of the accompanying video. These instructions are often generated using simple templates, commonly in a question-answer format. Here's an illustrative example with the underlined part generated as a pseudo instruction:

Question: ### Human: &lt;Video&gt; [Video Tokens] &lt;/video&gt; [Video Description] Is there any anomaly in the video?

Answer: ### Assistant: Yes, there are anomalies from 1.5s to 2s.

In the Question, [Video Tokens] denoted the tokens (places) for inserting visual prompts. [Video Description] is simple video clip details, e.g., video length and frame sample rate. During the instruction-tuning of VAD-LLaMA, the Question is first transformed into textual embeddings with a pre-trained LLM (vicuna-7b [26]) and then concatenated with visual prompts to serve as the inputs of LLaMA. Later, the textual embeddings transformed from the Answer are utilized as the "ground truth" of LLaMA's generation.

## 3.2. Model Training

The training pipeline of VAD-LLaMA is outlined in Figure 3. We implement a three-phase approach. In the first phase, clip-level VE features are input into the VADor to establish a baseline for predicting initial anomaly scores. In the second phase, these preliminary scores facilitate the aggregation of representative normal and abnormal features within the LTC module. This module is co-trained with the VADor to merge long-term contextual information into the process of representation learning and anomaly detection. In the third phase, we refine our VAD-LLaMA model by exclusively training the feature adaptor (the projection layer), utilizing the robust features produced by the VE and LTC modules. These features impart a broad understanding of general video content and specific guidance for anomaly detection, which are integral to the instruction-tuning process. Concurrently, the improved anomaly scores, ascertained through the VADor in conjunction with the LTC module, are transformed into pseudo instructions. These are then amalgamated with straightforward text templates, serving as the instruction-tuning data for LLaMA. The details of the training phases are given below.

Phase 1: Training VADor. In this phase, we train a simple VADor baseline by directly passing the VE features through AP g, as shown in the left of Figure 3. Facing the scarcity of anomalous data and labels in VAD, many researchers opt to address the VAD problem in a Weakly Supervised Video Anomaly Detection (WSVAD) framework. In this setting, each training video is annotated with a binary anomaly label y ∈ {0 , 1} (i.e., normal or abnormal), denoting whether it is categorized as normal or abnormal. This allows for training VAD models without the need for frame-level annotations of specific anomalous events, making it more feasible in real-world applications. In this work, we adopt the same setting as in WSVAD.

The prevailing approach to WSVAD is Multiple Instance Learning (MIL). It aims to train a clip-level AP g base on the VE features {xi} m i=1 . In this process, it distinguishes the most anomalous clip in a normal video (i.e., y = 0) as normal, and identifies the most anomalous clip within an abnormal video (i.e., y = 1) as abnormal. To achieve this, MIL constructs a tuple set S, one tuple for each video, which includes the prediction y ′ generated by g on the most anomalous snippet and the corresponding video-level label, denoted as y. This tuple is represented as (y ′ , y), where y ′ is computed as max{g(xi)} m i=1 . The parameters of g are trained by minimizing the binary cross-entropy (BCE) loss:

<!-- formula-not-decoded -->

In this way, for a normal video with y = 0, by minimizing max{g(xi)} m i=1, g is compelled to assign low abnormal probabilities to all video clips. Conversely, for an abnormal video with y = 1, by maximizing max{g(xi)} m i=1, g is trained to yield an even higher probability for the most confident abnormal snippet. Following previous method [13] that directly predicts binary logits for the normal and abnormal probabilities. During the inference of WSVAD, we utilize the abnormal probabilities as anomaly scores to calculate the evaluation metrics (AUC).

Phase 2: Co-Training VADor and LTC. The VADor trained in phase 1 is a MIL-based baseline. It is trained from VE features. However, this VE was pre-trained on short videos whose contexts that significantly differ from those in long and complex videos of VAD.

To enhance the long-term video representation ability of VADor, we co-train it with a novel LTC module, which is designed to encode the most normal as well as the most abnormal events seen in the input video. As mentioned in Sec. 3.1, the top-K normal and abnormal VE features are collected and stacked into a normal list N and an abnormal list A, respectively. In the forward pass, the top-K selection is based on the preliminary anomaly scores predicted from the VADor baseline, we pick up the VE features with the K lowest (highest) scores as the items of list N (A). Moreover, we online update the lists by re-collecting the historical features based on their anomaly scores, when inputting a new video clip into the LTC module.

Based on the "memory" stored in the LTC module, the next question is how to incorporate them into the representation (i.e., the VE features) of new video clips. To this end, we introduce the cross-attention mechanism to automatically retrieve contextual features from the LTC lists, based on their relevance to the current VE feature. Specifically, we regard the current VE feature xi as the query and the stacked features from LTC lists N , A are utilized as the key and value at the same time. Taking the i-th VE feature as an example, the process derives as:

<!-- formula-not-decoded -->

we denote the acquired feature from Eq. (2) as ni , ai, separately from N and A. Here, × denotes the dot product. In detail, the i-th VE feature is first multiplied with the LTClisted features to generate attention weights, then the relative features are retrieved with these weights, i.e. the higher the feature similarity, the higher the attention weight. After that, we combine these features with the VE feature xi:

<!-- formula-not-decoded -->

here, instead of carefully tuning hyper-parameters to manually select a descent weight tuple, we introduce neural soft weights with parameters w n , wa ∈ W to automatically balance the features. Then the feature after cross-attention is fed into the AP and form a more robust VADor. Later, the VADor and the LTC are co-trained and supervised with BCE loss as in (1).

Note that, to further integrate the short-term historical information that involves the variation of happening events, we add a list for storing the past K VE features, represented as H = {hj} K j=1 . These VE features in the short-term history contribute to learning a more comprehensive feature representation and boost the robustness of VADor. In this way, we upgrade the LTC module with a plus version, namely Long-Short-Term Context (LSTC) module. Extensive experiments demonstrate the effectiveness of VADor with the LTC and LSTC module as in Sec. 4.5 .

Phase 3: Instruction-Tuning Adaptor. In this phase, we incorporate the VADor with the pre-trained VE and LLaMA from Video-LLaMA [24] by adding an adaptor. Considering the limited training data in VAD, we opt to freeze the large modules (VADor, VE, and LLaMA) and train only the adaptor that aligns the feature distribution of VADor with the LLaMA. The frozen modules helps to reduce the model's dependence on the scale of the training data. Also the features from the well-trained VADor provide a comprehensive understanding of general video content and specific guidance for anomaly detection, which are integral to the instruction-tuning process.

Anomaly Prompt. To seamlessly incorporate video representations and anomaly information into LLaMA, we utilize the LTC feature x˜ ˜ i trained in the co-training phase as the anomaly prompt. As illustrated in Figure 2, the anomaly prompt is stacked with the VE feature, resulting in xˆ ˆ i = [xi , x ˜ i ]. Then, we add an adaptor (linear layer) f to project them into the same dimension as the inputs of LLaMA. The output feature embedding f(xˆ ˆ i ) serves as clip-level soft visual prompts that guide the pre-trained LLaMA to generate text that is conditioned on the visual content and anomaly status of the video clip.

Pseudo-Instruction. There is no frame-level anomaly annotation in WSVAD, so it is not intuitive how to construct temporal instructions for LLaMA. To address this challenge, we propose to convert anomaly scores (output by VADor) into pseudo instructions and manually compose anomaly-related text templates to generate instruction-tuning data. The process is showcased in the red line of Figure 2 and Figure 3 .

To generate video instruction pairs for VAD data (e.g., UCF-Crime dataset [18]), we start by inserting the visual prompts {xˆ ˆ } m i=1 into the textual embeddings Q of the Question. Then, we convert the predicted anomaly score from the VADor into the pseudo instruction as showcased in the Answer , e.g., "0.9" is transformed into the underlined part of Answer in Sec. 3.1, according to the time duration of the video clip. Hence, for i-th video clip, the video instruction pair becomes (ˆy ′ , y ˆ ), here yˆ ˆ ′ = [f(xˆ ˆ i ), Q] stands for the inputs of LLaMA and yˆ ˆ denotes the textual embeddings transformed from the pseudo instruction in Answer .

To prevent overfitting on VAD videos, we incorporate a diverse training sample set P, drawing from both the UCFCrime and WebVid datasets. Finally, we train the adaptor with the cross-entropy loss. Note that Cross-Entropy (CE) loss is commonly employed for training LLMs [2], which quantifies the disparity between the text sequence generated by the model and the target text sequence. The formula of CE loss is derived as follows:

<!-- formula-not-decoded -->

where n is the number of embedding tokens in yˆ ˆ , y ˆ j is the true label for token j and LLaMA(ˆy ′ )j is the LLaMApredicted probability for token j. Additionally, we illustrate the detailed pipeline with a pseudo-code in the Appendix.

## 4. Experiments

## 4.1. Datasets and Evaluation Metrics

To verify the performance of our VADor, we conducted extensive experiments and ablations on two standard WSVAD evaluation datasets [14 , 18]. As per the standard in WSVAD, the training videos only have video-level labels, and the test videos have frame-level labels. Other details of the experimental setting are given below.

UCF-Crime [18] is a large-scale dataset comprising 1,900 untrimmed real-world surveillance videos on general scenarios, encompassing both outdoor and indoor environments. The dataset boasts a total duration of 128 hours and includes 13 distinct classes of anomalous events. It is divided according to the standard split, with a training set of 1,610 videos, and a test set of 290 videos.

TAD dataset [14] features real-world traffic scene videos, with an average of 1,075 frames per video. These videos encompass over seven common road-related anomaly categories. The dataset is split into a training set consisting of 400 videos and a test set comprising 100 videos.

Evaluation Metrics. Following previous works [13 , 18], we adopted the Area Under the Curve (AUC) of the framelevel ROC (Receiver Operating Characteristic) as the main WSVAD evaluation metric for TAD and UCF-Crime. Intuitively, a larger AUC means a larger margin between the normal and abnormal predictions of video clips, suggesting a superior anomaly classifier. Taking inspiration from UMIL [13], our evaluation goes beyond calculating AUC for the entire test set, denoted as AUCO. We also compute the AUC specifically for abnormal videos, referred to as AUCA. It is for excluding normal videos where all clips are normal (label 0), and keeping only the abnormal ones with both kinds of clips (label 0,1). This selective evaluation truly challenges a classifier's capability to accurately localize and detect anomalies within a mixed context.

## 4.2. Implementation Details

Given that VAD videos are predominantly sourced from CCTV surveillance, which typically lacks audio signals, we omit the audio branch while retaining the visual part as the VE for generating video clip features. We also implemented the pre-trained LLaMA from Video-LLaMA [24] to retain general video description knowledge. In our VAD-LLaMA, two fc layers are used to implement the AP g, and another two fc layers are used to balance the features as W in the LTC module, each corresponding to a feature list, i.e., normal or abnormal. All the fc layers in VAD-LLaMA are initialized with random weights and trained to locate and describe the possible anomalies in videos. The length of LTC lists is uniformly set as 4 by default, according to the ablation study in Sec. 4.5 .

We trained our model with the AdamW optimizer [11] using an initial learning rate of 1e-5, weight decay of 0 . 001 , and batch size of 8 in the training phases 1 and 2. Due to the large memory consumption of LLaMA, the batch size was reduced to 2 during phase 3. We utilized the cosine annealing scheduler and warmed up the learning rate for 5 epoch among each training phase. The VADor baseline was trained with MIL-based BCE loss for 30 epochs, followed by 30 epochs of co-training with the LTC module. After that, we trained the adaptor for 30,000 iterations and froze the VADor, VE, and LLaMA. We conducted all experiments on 4 Nvidia L40 GPUs. We implemented the max value scores and max margin scores in Eq 1 as [13 , 14].

## 4.3. Quantitative Results

Weakly Supervised Video Anomaly Detection (WSVAD). In Table 1, we compared our VADor with other state-of-the-art (SOTA) methods in WSVAD. On UCFCrime [18], VADor with LTC achieves the best AUCO and AUCA among all the methods, with an improvement of +0.88% and +2.44%, respectively. VADor with LTC achieves the second best AUC O in TAD [14] and significantly outperforms all methods on AUCA by +3.21%. Moreover, with the introduction of LSTC, we witness a further improvement among the two benchmarks.

Overall Observations. 1) Notice that our baseline VADor performs far better than the previous MIL-based two-stage model [18]. This validates the strong video representation power of the pre-trained VE [24]. 2) Moreover, our VADor with LTC module significantly improves the AUCA over VADor baseline (e.g., +4.45% on UCF and +10.84% on TAD), which demonstrates the effectiveness of incorpo-

Table 1. WSVAD comparison on UCF-Crime. "2-stage" and "E2E" stand for the two-stage pipeline and the end-to-end framework. "w" and "w/o" are abbreviations for "with" and "without". AUCO and AUCA denote that the AUC computed on the overall test set and only abnormal test videos, respectively. The best results are in bold, and the second-best results are underlined.

| Category    | Method              |   AUCO (%)  | AUCA (%)   |
|-------------|---------------------|-------------|------------|
|             | Sultani et al. [18] |       75.41 | 54.25      |
|             | Zhang et al. [25]   |       78.66 | -          |
|             | Motion-Aware [29]   |       79.1  | 62.18      |
|             | GCN-Anomaly [27]    |       82.12 | 59.02      |
|             | Wu et al. [21]      |       82.44 | -          |
|             | RTFM [19]           |       84.3  | -          |
|             | WSAL [14]           |       85.38 | 67.38      |
|             | ECUPL [23]          |       86.22 | -          |
| E2E         | UMIL [13]           |       86.75 | 68.68      |
| E2E         | VADor w/o LTC       |       85.9  | 66.67      |
| E2E         | VADor w LTC         |       87.63 | 71.12      |
| E2E         | VADor w LSTC        |       88.13 | 72.54      |

Table 2. WSVAD comparison on TAD benchmark.

| Category    | Method              |   AUCO (%)  | AUCA (%)   |
|-------------|---------------------|-------------|------------|
|             | Sultani et al. [18] |       81.42 | 55.97      |
|             | Motion-Aware [29]   |       83.08 | 56.89      |
|             | GIG [12]            |       85.64 | 58.65      |
|             | RTFM [19]           |       89.61 | -          |
|             | WSAL [14]           |       89.64 | 61.66      |
|             | ECUPL [23]          |       91.66 | -          |
| E2E         | UMIL [13]           |       92.93 | 65.82      |
| E2E         | VADor w/o LTC       |       85.2  | 58.19      |
| E2E         | VADor w LTC         |       90.91 | 69.03      |
| E2E         | VADor w LSTC        |       91.77 | 70.78      |

rating long-range contextual information into the anomaly analysis. Additionally, the incorporation of short-term historical information leads to a further enhancement of the AUC performance. 3) Our VADor achieves the second best AUC O results on TAD, it is mainly because we froze the pre-trained VE, while UMIL [13] fine-tuned the feature backbone with VAD data. The higher AUCO, but lower AUCA of UMIL demonstrate that UMIL is better at distinguishing the normal clips in normal videos, but our VADor achieves a better anomaly localization performance among anomalous videos with a much higher AUCA (+3.21%).

## 4.4. Quantitative Examples

We showcase a video example of 'Abuse' for comparison between our VAD-LLaMA and Video-LLaMA [24] in Figure 4. As observed, the Video-LLaMA, available as an open-source model, struggles to precisely correlate the detected anomaly (Abuse) with specific events in the video, notably the incident where a woman is knocked down. Additional examples of our model with examples are put in

Figure 4. An abuse example for comparison between the VADLLaMA and Video-LLaMA. The red boxes in the frames are ground-truth anomalies. The orange boxes are the question from humans. The gray and blue boxes are the answers from the VideoLLaMA and our VAD-LLaMA, respectively. Best viewed in color.

![Image](artifacts/image_000003_42469625957f716d3939dde17671651e85e60bdc305d5dfd8ab9deef63ee53c0.png)

Figure 5. The model effectively identifies anomalies, i.e ., "Abuse" and "Car accident", accurately pinpoints their temporal locations, and provides a detailed description of the anomalies. For normal videos, our VAD-LLaMA is able to comprehensively analyze the content of the video and eliminate the possibility of anomalies. Moreover, users are able to engage in multi-turn dialogues pertaining to the video content. More qualitative examples and comparisons between VAD-LLaMA and Video-LLaMA are moved to the Appendix.

## 4.5. Ablation studies

LTC Components. We validate the effectiveness of longterm context modeling in Table 3 with AUCO. By comparing the third and fourth lines with the first line, we observe that the normal (abnormal) features in long-range video context can improve AUCO from 85.90% to 87.08% (87.45%) on UCF-crime and 85.20% to 88.39% (89.08%) on TAD. In the fifth line, with the combination of the normal and abnormal contexts, a further improvement of AUCO proves the critical role of the long-range video context in robust anomaly mining. In addition, we verify the success of short-term historical information in VADor, which boosts the AUC O to reach 88.13% on UCF-crime and 91.77% on TAD. For an independent evaluation of the effectiveness of our VADor, we re-implemented the previous SOTA UMIL [13] using the VE features, denoted as UMIL*. The results are presented in line 2. Our VADor with LTC in line 4 consistently outperforms UMIL* (+0.85% on UCF-Crime and +2.26% on TAD), thereby validating the efficacy of our design, based on the same feature backbone.

Figure 5. Two qualitative examples of VAD-LLaMA.

![Image](artifacts/image_000004_59f965ac2e5285fe0ca89a09c7d9dcbd47133d48c9748fd1f0ec603c233b4876.png)

Table 3. Ablation studies of the components in the LTC module. Here, "Nor" and "Abn" denote the normal and abnormal list, respectively. "His" stands for the short-term history list and "UMIL" is the unbiased term proposed in [13].

| Baseline    | Nor    | Abn    | His    | UMIL    |   UCF  |   TAD |
|-------------|--------|--------|--------|---------|--------|-------|
| ✓           |        |        |        |         |  85.9  | 85.2  |
| ✓           |        |        |        | ✓       |  86.78 | 88.65 |
| ✓           | ✓      |        |        |         |  87.08 | 88.39 |
| ✓           |        | ✓      |        |         |  87.45 | 89.08 |
| ✓           | ✓      | ✓      |        |         |  87.63 | 90.91 |
| ✓           | ✓      | ✓      | ✓      |         |  88.13 | 91.77 |

LTC Length. In the LTC, K is employed as the length of the feature lists. Through empirical analysis presented in Table 4, we determine that K = 4 is a suitable choice across the two datasets, and thus, it is the default setting for our experiments. In general, the selection of K hinges on the reliance of anomaly detection on video contexts. For instance, a small K might not capture sufficient temporal in- formation, while a large K could involve unexpected noise.

Class-wise AUC. On the UCF-Crime dataset, each test video is labeled with the class of anomaly, enabling us to analyze models' capabilities in detecting subtle abnormal events through class-wise AUCA comparisons. In Figure 6 , we compare VADor with the baseline and UMIL, where "Average" represents the overall AUCA, and the remaining bars show the class-wise values.

Our observations are as follows: 1) Both the VADor baseline and UMIL demonstrate strong performance on

Figure 6. Class-wise AUCA of three methods on UCF-Crime. Here, "VADor" stands for our VADor with the LTC module.

![Image](artifacts/image_000005_ce761569716d3ac4c0e2ee03aaf63a8132a17b00d401f6707bf1f6d8cc6c454f.png)

Table 4. Ablation of the LTC Length on UCF-Crime and TAD.

| Threshold(%)    |   0  |    2  |    4  |    6  |     8 |
|-----------------|------|-------|-------|-------|-------|
| AUCO (%) - UCF  | 85.9 | 87.49 | 87.63 | 87.27 | 87.18 |
| AUCO (%) - TAD  | 85.2 | 90.18 | 90.91 | 90.65 | 90.13 |

anomaly classes characterized by drastic motions, such as "Assault" and "Vandalism". These classes represent intuitive anomalies primarily relying on feature representation learning in a short time duration, given that VE and the backbone of UMIL are sufficient to capture local details in short video clips. 2) However, these methods struggle to distinguish anomalies that depend on long-range temporal analysis, like "Arson" and "Shoplifting". These classes correspond to the hard examples, which are inadequately addressed by the long-term context modeling in our VADor. VADor with the LTC module performs similarly well on the aforementioned intuitive anomaly classes and significantly outperforms the other methods on other anomalies that require comprehensive context modeling. This substantial improvement contributes to the superior anomaly detection performance. Overall, observations 1 and 2 empirically validate the effectiveness of mining long-range video contexts for a more robust anomaly analysis.

## 5. Conclusion

In this work, we introduced VAD-LLaMA, a novel Video Anomaly Detection (VAD) approach that integrates videobased large language models (VLLMs) into the VAD framework, making the VAD model free from thresholds and able to explain the reasons for the detected anomalies. In our model, we introduced a Long-Term Context (LTC) module to mitigate the incapability of existing VLLMs in longrange context modeling. In addition, our three-phase training method significantly improves the efficiency of training VLLMs in specific domain as VAD by minimizing the requirements for VAD data and reducing the costs of annotating instruction-tuning data. Our approach was empirically validated by the state-of-the-art performance and extensive ablations on standard WSVAD benchmarks. Also, we showcased the anomaly localization and description capability of VAD-LLaMA in the multi-dialogue based on the video content. In the future, we seek to develop a VAD model with fast adaption capability that can detect new anomalies based on either a few example clips or textual descriptions of the targeted anomalies.

## References

- [1] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisser- ¨ ¨ man. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, pages 1728–1738, 2021. 2 , 3
- [2] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-and-language transformers. In CVPR , 2022. 6
- [3] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In CVPR, 2023. 3
- [4] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. ArXiv, 2023. 3
- [5] Chengkun He, Jie Shao, and Jiayu Sun. An anomalyintroduced learning method for abnormal event detection. Multimedia Tools and Applications, 2018. 2
- [6] Zijie Huang and Yulei Wu. A survey on explainable anomaly detection for industrial internet of things. In DSC, 2022. 1
- [7] Shunsuke Kamijo, Yasuyuki Matsushita, Katsushi Ikeuchi, and Masao Sakauchi. Traffic monitoring and accident detection at intersections. ITITS, 2000. 1
- [8] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ArXiv , 2023. 3
- [9] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. ArXiv, 2023. 2 , 3
- [10] Shuo Li, Fang Liu, and Licheng Jiao. Self-training multisequence learning with transformer for weakly supervised video anomaly detection. AAAI, 2022. 2
- [11] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. ICLR, 2019. 6
- [12] Hui Lv, Chunyan Xu, and Zhen Cui. Global information guided video anomaly detection. In ACM MM, 2020. 7
- [13] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. Unbiased multiple instance learning for weakly supervised video anomaly detection. In CVPR, 2023. 1 , 3 , 5 , 6 , 7 , 8
- [14] Hui Lv, Chuanwei Zhou, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang. Localizing anomalies from weakly-labeled videos. TIP, 2021. 1 , 2 , 6 , 7
- [15] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video

understanding via large vision and language models. ArXiv , 2023. 1

- [16] Sadegh Mohammadi, Alessandro Perina, Hamed Kiani, and Vittorio Murino. Angry crowds: Detecting violent events in videos. In ECCV, 2016. 1
- [17] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. ArXiv, 2023. 3
- [18] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In CVPR, 2018. 1 , 2 , 5 , 6 , 7
- [19] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In ICCV, 2021. 1 , 2 , 7
- [20] Peng Wu and Jing Liu. Learning causal temporal relation and feature discrimination for anomaly detection. TIP, 2021. 2
- [21] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In ECCV, 2020. 7
- [22] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection. In ECCV, 2020. 2
- [23] Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang. Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection. In CVPR, 2023. 7
- [24] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. ArXiv, 2023. 1 , 2 , 3 , 5 , 6 , 7
- [25] Jiangong Zhang, Laiyun Qing, and Jun Miao. Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection. In ICIP, 2019. 2 , 7
- [26] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. ArXiv, 2023. 3 , 4
- [27] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection. In CVPR, 2019. 2 , 7
- [28] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. AAAI, 2023. 1
- [29] Yi Zhu and Shawn Newsam. Motion-aware feature for improved video anomaly detection. BMVC, 2019. 2 , 7