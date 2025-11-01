---
title: 'AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection'
type: method
categories:
- Weakly Supervised
- Hybrid
github_link:
description: A novel weakly supervised framework leveraging audio-visual 
  collaboration to improve the robustness and accuracy of video anomaly 
  detection.
benchmarks:
- xd-violence
- ucf-crime
- shanghaitech
authors:
- Peng Wu
- Wanshun Su
- Guansong Pang
- Yujia Sun
- Qingsen Yan
- Peng Wang
- Yanning Zhang
date: '2023-10-01'
---

## AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection

Peng Wu, Wanshun Su, Guansong Pang Member, IEEE, Yujia Sun, Qingsen Yan Member, IEEE , Peng Wang Member, IEEE and Yanning Zhang Fellow, IEEE

Abstract—With the increasing adoption of video anomaly detection in intelligent surveillance domains, conventional visualonly detection approaches often struggle with information insufficiency and high false-positive rates in complex environments. To address these limitations, we present a novel weakly supervised framework that leverages audio-visual collaboration for robust video anomaly detection. Capitalizing on the exceptional cross-modal representation learning capabilities of Contrastive Language-Image Pretraining (CLIP) across visual, audio, and textual domains, our framework introduces two major innovations: an efficient audio-visual fusion that enables adaptive crossmodal integration through lightweight parametric adaptation while maintaining the frozen CLIP backbone, and a novel audiovisual prompt that dynamically enhances text embeddings with key multimodal information based on the semantic correlation between audio-visual features and textual labels, significantly improving CLIP's generalization for the video anomaly detection task. Moreover, to enhance robustness against modality deficiency during inference, we further develop an uncertaintydriven feature distillation module that synthesizes audio-visual representations from visual-only inputs. This module employs uncertainty modeling based on the diversity of audio-visual features to dynamically emphasize challenging features during the distillation process. Our framework demonstrates superior performance across multiple benchmarks, with audio integration significantly boosting anomaly detection accuracy in various scenarios. Notably, with unimodal data enhanced by uncertaintydriven distillation, our approach consistently outperforms current unimodal VAD methods.

Index Terms—video anomaly detection, audio-visual collaboration, weakly supervised learning.

## I. INTRODUCTION

V IDEO anomaly detection (VAD), as a pivotal technology in intelligent surveillance systems, focuses on identifying anomalous events within videos and has attracted substantial research interest in recent years [1]–[11]. Due to the rarity of anomalies and the high cost of manual annotation, fully supervised frameworks are impractical for large-scale deployment. As a solution, weakly supervised video anomaly detection (WSVAD) methods [12]–[15] have gained traction, aiming to discover latent anomalies under coarse supervision. Current WSVAD methods primarily rely on the multiple

Peng Wu, Wanshun Su, Qingsen Yan, Peng Wang, and Yanning Zhang are with the School of Computer Science, Northwestern Polytechnical University, China. E-mail:{xdwupeng, suws0616, qingsenyan}@gmail.com; {peng.wang, ynzhang}@nwpu.edu.cn.

Guansong Pang is with the School of Computing and Information Systems, Singapore Management University, Singapore. E-mail: pangguansong@gamil.com.

Yujia Sun is with the School of Artifical Intelligence, Xidian University, China. E-mail: yjsun@stu.xidian.edu.cn.

Manuscript received April 19, 2021; revised August 16, 2021.

Fig. 1. Left: Illustration of audio-visual collaboration effects; Right: Illustration of our proposed distillation (UKD) effects.

![Image](artifacts/image_000000_e9764a03f6873fc9ea7586f2dc6faa2fccd69b228e938f53e7d376bfa27fc58e.png)

instance learning (MIL) framework, using video-level labels for model training [12], [16]. Specifically, these approaches treat videos as bags of segments (instances) and distinguish anomalous patterns through the hard attention mechanism (a.k.a Top-K) [17]. With the rapid advancement of foundation models, Contrastive Language-Image Pretraining (CLIP) [18] has shown remarkable potential in various downstream tasks, including video understanding [19], [20]. Building on the remarkable success of CLIP, recent methods like VadCLIP [21] and TPWNG [15] have advanced WSVAD by leveraging CLIP's semantic alignment capabilities.

However, these methods, whether CLIP-based or conventional, predominantly rely on unimodal visual information, which often leads to significant detection limitations in complex real-world scenarios. Visual occlusion, extreme lighting variations, and environmental noise can render visual features unreliable or ambiguous [22]–[24]. In these challenging conditions, multimodal information, particularly audio, offers indispensable contextual cues that can complement and enhance visual-based detection. For instance, audio remains robust when visual data is compromised, allowing detection of off-camera events. In acoustically rich environments, certain anomalies like explosion, scream, or gunshot exhibit distinct acoustic signatures, making them more discriminative in the audio domain. Similarly, in low-light conditions where visual features degrade, audio serves as a critical supplementary modality. These observations underscore the importance of integrating audio and video modalities, as their complementary nature can significantly enhance the accuracy and robustness of anomaly detection systems in diverse and challenging environments. We illustrate the impact of audio-visual integration for WSVAD in Figure 1.

Existing attempts [22], [25], [26] to incorporate audio into video anomaly detection typically adopt traditional feature concatenation methods, such as fusing visual features extracted by I3D [27] or C3D [28] with audio features extracted by VG-

Gish [29]. These approaches fail to fully exploit the potential of multimodal learning, resulting in suboptimal cross-modal integration. Moreover, they overlook the inherent semantic alignment between visual and auditory modalities, which are essential for enhancing anomaly detection performance.

To address these limitations, we propose AVadCLIP, a WSVAD framework that leverages audio-visual collaborative learning to drive audio-visual anomaly detection by CLIPpowered cross-modal alignment. AVadCLIP fully exploits CLIP's intrinsic capability to establish semantic consistency across vision, text, and audio, ensuring that video anomaly detection is performed within a unified multimodal semantic space rather than merely fusing raw features. Our framework introduces three significant innovations: an efficient audiovisual feature fusion mechanism that is different from the naive feature concatenation and achieves adaptive cross-modal integration through lightweight parametric adaptation while keeping the CLIP backbone frozen; a novel audio-visual prompt mechanism dynamically enriches text label embeddings with key multimodal information, enhancing contextual understanding of videos and enabling more precise identification of different categories; and an uncertainty-driven feature distillation (UKD) module that generates audio-visual-like enhanced features in audio-missing scenarios, ensuring robust anomaly detection performance (as illustrated in Figure 1). Overall, our AVadCLIP relies on only a small set of trainable parameters, effectively transferring CLIP's pretrained knowledge to the weakly supervised audio-visual anomaly detection task. Furthermore, by employing a distillation strategy based on data uncertainty modeling, we further transfer the learned knowledge from our audio-visual anomaly detector to a unimodal detector, enabling robust anomaly detection in scenarios with incomplete modalities.

In summary, our main contributions are as follows:

- We propose a WSVAD framework that harnesses audio-visual collaborative learning, leveraging CLIP's multimodal alignment capabilities. By incorporating a lightweight adaptive audio-visual fusion mechanism and integrating audio-visual information through promptbased learning, our approach effectively achieves CLIPdriven robust anomaly detection in multimodal settings.
- We design an uncertainty-driven feature distillation module, which transforms deterministic estimation into probabilistic uncertainty estimation. This enables the model to capture feature distribution variance, ensuring robust anomaly detection performance even with unimodal data.
- Extensive experiments on two WSVAD datasets demonstrate that our method achieves superior performance in audio-visual scenarios, while maintaining robust anomaly detection results even in audio-absent conditions.

## II. RELATED WORK

## A. Video Anomaly Detection

Video anomaly detection has been extensively studied in recent years, with existing approaches broadly categorized into semi-supervised and weakly supervised methods. Among them, semi-supervised methods primarily rely on normal video clips for training and identify anomalies by detecting deviations from learned normal patterns during inference. These methods commonly adopt self-supervised learning techniques [30]–[32], such as reconstruction [33], [34] or prediction [35], [36]. Reconstruction-based methods assume that the model can effectively reconstruct normal videos, whereas abnormal videos, due to distributional discrepancies, result in significant reconstruction errors. Autoencoders [37], [38] are widely employed to capture normal pattern features, with reconstruction error serving as an anomaly indicator. Predictionbased methods [39] utilize models to forecast future frames, detecting anomalies based on prediction errors. However, a key limitation of semi-supervised methods is their tendency to overfit normal patterns, leading to poor generalization to unseen anomalies.

Weakly supervised methods, in contrast, typically adopt the MIL framework, requiring only video-level anomaly labels and significantly reducing annotation costs. The classic work, DeepMIL [12], which employs a ranking loss to distinguish normal from anomalous instances. Furthermore, two-stage self-training strategy has been proposed to further enhance detection, where high-confidence anomalous regions identified during MIL training serve as pseudo-labels for a secondary refinement phase [40]–[42]. With the rise of Vision-Language Models (VLMs) [43], CLIP has shown remarkable crossmodal capabilities and is increasingly applied to WSVAD. VadCLIP [21], the first CLIP-based WSVAD method, integrates textual priors via text and visual prompts, enhancing anomaly detection. Building on this, TPWNG [15] refines feature learning through a two-stage approach. Recent research trends focus on large model driven strategies, e.g., training-free frameworks [44], [45], spatiotemporal anomaly detection [46], and open-scene anomaly detection [47]. Recent advances in multi-modal fusion [48] introduce powerful frameworks combining diverse modalities such as visual and audio features. For instance, AVCL [49] and DSRL [50] have shown significant promise in improving anomaly detection by leveraging both visual and audio cues.

## B. Audio-Visual Learning

The integration of audio and visual information has emerged as a critical research direction in multimodal learning, as it not only enhances model performance but also facilitates a deeper understanding of complex scenes. Significant progress has been achieved in various aspects of audio-visual fusion [51], [52]. In audio-visual segmentation, researchers aim to accurately segment sound-producing objects based on audio-visual cues. Chen et al. [53] proposed a novel informative sample mining method for audio-visual supervised contrastive learning. Ma et al. [54] introduced a two-stage training strategy to address the audio-visual semantic segmentation (AVSS) task. Building on these works, Guo et al. [55] introduced a new task: Open-Vocabulary AVSS (OV-AVSS), which extends AVSS to open-world scenarios beyond predefined annotation labels. Audio-visual event localization aims to identify the spatial and temporal locations of both visual and auditory events, with attention mechanisms widely used for modality fusion. For

instance, He et al. [56] proposed an audio-visual co-guided attention mechanism, while Xu et al. [57] introduced an audioguided spatial-channel attention mechanism. Related tasks include audio-visual video parsing [58], [59] and audio-visual action recognition [60]. Audio-visual anomaly detection [25], [61] has also become a growing research hot. For example, Yu et al. [62] applied a self-distillation module to transfer singlemodal visual knowledge to an audio-visual model, reducing noise and bridging the semantic gap between single-modal and multimodal features. Similarly, Pang et al. [63] proposed a weighted feature generation approach, leveraging mutual guidance between visual and auditory information, followed by bilinear pooling for effective feature integration.

## C. Large Models in Video Understanding

In recent years, large models have exhibited exceptional capabilities in perception and reasoning for video understanding tasks, significantly accelerating the shift from purely visual models to multimodal video understanding frameworks. Representative visual models, such as VideoMAE [64], employ masked self-supervised learning to effectively model spatiotemporal dynamics in videos, facilitating their widespread application in video classification, action recognition, and anomaly detection. With the success of VLMs like CLIP [18] and ALIGN [65], integrating language priors into video understanding has emerged as a prominent research trend. These models perform cross-modal semantic alignment through joint image-text encoding and have been widely adopted in tasks such as zero-shot action recognition, video retrieval, and openvocabulary scene understanding. Further advances, including X-CLIP [66] and VideoCLIP [20], introduce temporal modeling into VLM architectures, significantly improving semantic comprehension of long-form video content. Meanwhile, VLMbased video reasoning tasks are gaining increasing attention. Models such as VL-T5 [67] and VideoChat [68] leverage language-guided mechanisms to enable video question answering, event interpretation, and causal reasoning, thereby substantially broadening the scope of video understanding.

## III. METHODOLOGY

## A. Problem Statement

Given a training set of videos {Vi}, where each video V contains both visual and corresponding audio information, along with a video-level label y ∈ R C . Here, C indicates that the number of categories (including the normal class and various anomaly classes). To facilitate model processing, we employ a video encoder and an audio encoder to extract highlevel features X v ∈ R N×d and X a ∈ R N×d , respectively, where N represents the temporal length of the video (i.e., the number of frames or snippets) and d denotes the feature dimensionality. The objective of WSVAD task is to train a detector using all available X v, X a , and their corresponding labels from the training set, enabling the model to accurately determine whether each frame in a test sample is anomalous and to identify the specific anomaly category.

The overall pipeline of our method as shown in Figure 2, starts with extracting features from video and audio using dedicated encoders, then adaptively fuses them for multimodal correspondence learning. We combine a classification branch with a CLIP-based alignment approach, using a audio-visual prompt to inject fine-grained multimodal information into text embeddings. Additionally, an uncertainty-driven distillation is employed to improve anomaly detection robustness in scenarios with incomplete modalities.

## B. Video and Audio Encoders

Video encoder. Leveraging CLIP's robust cross-modal representation, we use its image encoder (ViT-B/16) as a video encoder, in contrast to traditional models like C3D and I3D, which are less effective in capturing semantic relationships. We extract features from sampled video frames using CLIP, but to address CLIP's lack of temporal modeling, we incorporate a lightweight temporal model, such as Graph Convolution Network (GCN) [16] and Temporal Transformer [21], to capture temporal dependencies. This approach ensures efficient transfer of CLIP's pre-trained knowledge to the WSVAD task. Audio encoder. For audio feature extraction, we use Wav2CLIP [69], a CLIP-based model that maps audio signals into the same semantic space as images and text. The audio is first converted into spectrograms, then sampled to match the number of video frames. These audio segments are processed by Wav2CLIP to extract features. To capture contextual relationships, we apply a temporal convolution layer [70], which models local temporal dependencies, preserving key dynamics within the audio modality.

## C. Audio-Visual Adaptive Fusion

In multimodal feature fusion, while both video and audio contain valuable semantic information, their importance often varies depending on the specific task. Inspired by human perception mechanisms [71], our approach follows a visioncentric, audio-assisted paradigm, where video features serve as the primary modality, and audio features complement and enhance visual information. To preserve the generalization capability of the original CLIP model in downstream tasks while avoiding the introduction of excessive trainable parameters, we design a lightweight adaptive fusion that integrates audio features without significantly increasing computational overhead. We present the structure of this fusion in Figure 3.

Specifically, given the video feature X v and audio feature X a , we first concatenate them to obtain a joint representation X a+v ∈ R N×2d , which is then processed by two projection networks to generate the adaptive weight and residual feature. The first projection network computes adaptive fusion weights W, which determine the contribution of audio at each time step [72]. This is achieved through a linear transformation followed by a sigmoid activation:

<!-- formula-not-decoded -->

The second projection network is responsible for residual mapping, which transforms X a+v into a residual feature Xr Xres that encodes the fused information from both modalities:

<!-- formula-not-decoded -->

Fig. 2. The pipeline of our proposed AVadCLIP. Our method supports both multimodal inputs and visual-only inputs via distillation, enabling robust video anomaly detection through the proposed UKD strategy. Throughout the entire framework, the pre-trained CLIP backbone remains fully frozen, with only a few modules being trainable. This design allows for efficient and lightweight adaptation of CLIP's knowledge to the specific task of audio-visual anomaly detection.

![Image](artifacts/image_000001_6c781f39ea7450bb1493fad6c37a5c023ae5220ce298d98266cfda2f996d0250.png)

Fig. 3. The pipeline of our proposed adaptive fusion module, binary classifier, visual enhancement network, and uncertainly modeling network.

![Image](artifacts/image_000002_261617b5586394102fcc581ebd0613184571c6924a878a2180948488d2b897c7.png)

Finally, the fused representation X av is obtained by adaptively incorporating the residual feature into the original video feature:

<!-- formula-not-decoded -->

where ⊙ denotes element-wise multiplication. The adaptive weight W dynamically adjusts the degree of audio integration, ensuring that video features remain dominant while audio features provide auxiliary information. Additionally, the residual mapping enhances the expressiveness of the fused representation by capturing nonlinear transformations. By introducing an adaptive fusion mechanism and maintaining a lightweight design, our fusion approach effectively balances efficiency and expressiveness, leveraging the complementarity of visual and audio modalities while minimizing computational overhead.

## D. Dual Branch Framework with Prompts

We leverage a dual-branch framework [21] for the WSVAD task, consisting of a classification branch and an alignment branch, which effectively leverage audio-visual information to improve detection accuracy. Classification branch consists of a lightweight binary classifier (as shown in Figure 3), which takes X av as input and directly predicts the framelevel anomaly confidence A. Alignment branch leverages the cross-modal semantic alignment mechanism, which computes the similarity between frame-level features and class label features. To obtain class label representations, we leverage the CLIP text encoder combined with the learnable textual prompt [73] and audio-visual prompt to extract class embeddings, ensuring unified semantic alignment between visual and textual modalities. Given a set of predefined class labels (e.g.,

"normal", "fighting"), we first introduce a learnable textual prompt, then we concatenate the textual prompt with class labels and feed them into CLIP text encoder to obtain the class representation X c . Compared to the manually defined prompt, the learnable prompt allows the model to dynamically adjust textual representations during training, making them more suitable for the specific requirements of WSVAD. Furthermore, we incorporate an audio-visual prompt into the class label features to enrich the class representations with additional multimodal information.

The proposed audio-visual prompt mechanism aims to dynamically inject instance-level key audio-visual information into text labels to enhance the representation. Specifically, we leverage the anomaly confidence A from the classification branch and audio-visual features X av to generate a video-level global representation:

<!-- formula-not-decoded -->

where Norm represents the normalization operation. Next, we calculate the similarity matrix Sp Sp between the class representation X c and the global representation Xp Xp to measure the alignment between class labels and videos:

<!-- formula-not-decoded -->

based on Sp Sp , we generate the enhanced instance-level audiovisual prompt X mp :

<!-- formula-not-decoded -->

This operation dynamically adjusts the class representation's focus on different video instances by calculating the similarity between global audio-visual features and class labels, thereby enhancing cross-modal alignment.

Then, we add X mp and the class representation X c , followed by a feed-forward network (FFN) transformation and a skip connection to obtain the final instance-specific class embedding X cp :

<!-- formula-not-decoded -->

where ADD represents element-wise addition.

This dual-branch framework provides anomaly confidence through the classification branch and refines category identification with class information via the alignment branch, improving robustness and enabling fine-grained anomaly detection.

## E. Optimization of Audio-Visual Model

For the classification branch, we adopt the Top-K mechanism, as proposed in previous work [25], to select the top K anomaly confidence values from both normal and abnormal videos, which are averaged as the video-level prediction. The classification loss LBCE is then computed using binary crossentropy between the prediction and groundtruth class.

In the case of the alignment branch, the MIL-Align mechanism [21] is applied. We compute an alignment map M , reflecting the similarity between frame-level features X av and all category embeddings X cp . For each row in M, the top K similarities are selected and their average is used to quantify the alignment between the video and the current class. This results in a vector S = {s1, . . . , s m } representing the similarity between the video and all possible classes. Then the multi-class prediction is then calculated as:

<!-- formula-not-decoded -->

where pi represents the prediction for the i th class, and τ is the temperature scaling parameter. Then, we compute LNCE based on cross-entropy. Besides, to address the class imbalance in WSVAD , where normal samples dominate and anomaly instances are sparse, we employ the focal loss [74]. Finally, the overall loss LALIGN for alignment branch is the average of LNCE and LF OCAL .

## F. Uncertainty-Driven Distillation

In the WSVAD task, audio serves as a complementary modality to video, enhancing detection accuracy. However, audio may be unavailable in practical scenarios, leading to performance degradation. To address this, we apply knowledge distillation by using a pre-trained multi-modal (video+audio) teacher model to guide a unimodal (video-only) student model, ensuring robust anomaly detection even without audio. Traditional knowledge distillation methods typically assume a deterministic transfer of knowledge, employing mean square error (MSE) loss to align the student model with the teacher's feature representations. However, this approach fails to account for the inherent uncertainty in audio-visual feature fusion. In real-world scenarios, factors such as noisy audio or occluded visual content can introduce distortions in the fused features, leading to inaccurate feature representations and diminished generalization capability.

To overcome this, we propose a probabilistic uncertainty distillation strategy [75], [76], which models data uncertainty during distillation, improving the student model's robustness across diverse scenarios. Specifically, assume Xav,i = Xvs,i+ ϵσi, where ϵ ∼ N (0 , I) , X vs represents enhanced visual features generated from the student model, and it is derived from X v after passing through a visual enhancement network, which is illustrated in Figure 3. Besides, σi refers to the inherent uncertainty between the i th pair of features. Then we model the observation as a Gaussian likelihood function to more accurately quantify data uncertainty in the feature distillation. The relationship between the audio-visual fusion feature X av,i and the unimodal feature X vs,i is formulated as:

<!-- formula-not-decoded -->

where θ is the parameter of models, to maximize the likelihood for each pair of features Xav,i and Xvs,i, we adopt the loglikelihood form:

<!-- formula-not-decoded -->

In practice, we design a network branch (a simple threelayer convolutional neural network, which is shown in Figure 3) to predict the variance σ 2 i and reformulate the likelihood

maximization problem as the minimization of a loss function. Specifically, we employ an uncertainty-weighted MSE loss:

<!-- formula-not-decoded -->

where L represents the number of feature pairs, and the constant term is omitted for clarity.

During the distillation process, the student model not only learns the unimodal feature X vs,i from the teacher model but also considers the feature uncertainty σ 2 i to optimize its learning strategy. Specifically, the first term of the loss function represents the feature similarity between the student and teacher models, normalized by σ 2 i . This assigns smaller weights to features with higher uncertainty, thereby avoiding overfitting to hard-to-learn information. The second term acts as a regularization term to prevent σ 2 i from becoming too small, ensuring effective distillation.

Ultimately, during the inference phase, we only input video and perform anomaly detection through the unimodal student model for audio-missing scenarios.

## IV. EXPERIMENTS

## A. Datasets and Evaluation Metrics

1) Datasets: We conduct extensive experiments on two audio-visual benchmarks: XD-Violence [25] and CCTVFightssub [49], both of which contain synchronized audio and visual modalities. Unlike traditional unimodal datasets, these benchmarks enable a more comprehensive evaluation of our framework's robustness under multimodal settings. XDViolence. As the largest publicly available audio-visual WSVAD dataset, XD-Violence [25] significantly surpasses existing datasets in scale and diversity. It comprises 3,954 training videos and 800 test videos, with the test set containing 500 violent and 300 non-violent videos. The dataset covers six distinct categories of violent events, including abuse, car accident, explosion, fighting, riot, and shooting, which occur at various temporal locations within videos.

CCTV-Fightssub . Derived from CCTV-Fights [77], CCTVFightssub [49] is a carefully curated subset designed to address audio-visual anomaly detection. The subset retains 644 highquality videos depicting real-world fight scenarios, each with meaningful audio content, making it a valuable resource for evaluating audio-visual anomaly detection methods in realworld surveillance contexts.

2) Evaluation Metrics: For performance evaluation, we adopt distinct metrics tailored to different granularities of WSVAD tasks. For coarse-grained WSVAD, we employ framelevel Average Precision (AP), which provides a comprehensive measure of detection accuracy across varying confidence thresholds. For fine-grained anomaly detection, we utilize mean Average Precision (mAP) [22] computed across multiple intersection over union (IoU) thresholds and the average mAP (AVG) across different thresholds. Specifically, we evaluate mAP at IoU thresholds ranging from 0.1 to 0.5 with an interval of 0.1, followed by reporting AVG across these thresholds.

TABLE I COARSE -GRAINED COMPARISONS ON XD-VIOLENCE .

| Method           | Reference    | Modality       |   AP(%) |
|------------------|--------------|----------------|---------|
| DeepMIL [12]     | CVPR 2018    | RGB(ViT)       |   75.18 |
| Wu et al. [25]   | ECCV 2020    | RGB(ViT)       |   80    |
| RTFM [78]        | ICCV 2021    | RGB(ViT)       |   78.27 |
| AVVD [22]        | TMM 2022     | RGB(ViT)       |   78.1  |
| Ju et al. [19]   | ECCV 2022    | RGB(ViT)       |   76.57 |
| DMU [26]         | AAAI 2023    | RGB(ViT)       |   82.41 |
| CLIP-TSA [79]    | ICIP 2023    | RGB(ViT)       |   82.17 |
| AnomalyCLIP [80] | CVIU 2024    | RGB(ViT)       |   78.51 |
| TPWNG [15]       | CVPR 2024    | RGB(ViT)       |   83.68 |
| VadCLIP [21]     | AAAI 2024    | RGB(ViT)       |   84.51 |
| AVadCLIP∗        | this work    | RGB(ViT)       |   85.53 |
| FVAI [63]        | ICASSP 202   | RGB(I3D)+Audio |   81.69 |
| MACIL-SD [62]    | ACMMM 20     | RGB(I3D)+Audio |   81.21 |
| CUPL [81]        | CVPR 2       | RGB(I3D)+Audio |   81.43 |
| AVCL [49]        | TMM 202      | RGB(I3D)+Audio |   81.11 |
| AVadCLIP         | this wor     | RGB(ViT)+Audio |   86.04 |

TABLE II COARSE -GRAINED COMPARISONS ON CCTV-FIGHTSsub .

| Method        | Reference    | Modality       |   AP(%) |
|---------------|--------------|----------------|---------|
| VadCLIP [21]  | AAAI 2024    | RGB(ViT)       |   72.78 |
| AVadCLIP∗     | this work    | RGB(ViT)       |   73.36 |
| MACIL-SD [62] | ACMMM20      | RGB(I3D)+Audio |   72.92 |
| DMU [26]      | CVPR202      | RGB(I3D)+Audio |   72.97 |
| AVCL [49]     | TMM 202      | RGB(I3D)+Audio |   73.2  |
| AVadCLIP      | this work    | RGB(ViT)+Audio |   73.38 |

## B. Implementation Details

We conduct experiments on an NVIDIA RTX 4090 GPU, where the visual enhancement network is a single-layer 1D convolutional network, which includes a convolutional layer with a kernel size of 3, padding size of 1, ReLU activation function, and a skip connection. Such an operation effectively facilitates the aggregation of local contextual information. For input processing, we employ a frame selection strategy tailored to different datasets, sampling one frame per 16 frames for XD-Violence and one frame per 4 frames for CCTV-Fightssub , using a uniform sampling strategy with a maximum frame count of 256; During optimization, we set the batch size, learning rate, and total epoch to 96, 1e − 5 , and 10, respectively.

## C. Comparison with State-of-the-Art Methods

1) Performance comparison on XD-Violence: Our experiments evaluate both coarse-grained and fine-grained anomaly detection performance on XD-Violence, comparing our AVadCLIP against state-of-the-art approaches, as shown in Tables I, III.

For coarse-grained anomaly detection, using only RGB inputs, AVadCLIP ∗ (∗ denotes RGB-only input) achieves an AP score of 85.53%, surpassing all existing vision-only methods. Notably, it outperforms VadCLIP, the previous best-performing RGB-only approach, by 1.0%, demonstrating superior visual anomaly detection. When incorporating audio, AVadCLIP further improves performance, significantly outperforming all

TABLE III FINE -GRAINED COMPARISONS ON XD-VIOLENCE .

| Method       | Reference    | Modality       |   mAP@IoU(%) |   mAP@IoU(%) |   mAP@IoU(%) |   mAP@IoU(%) |   mAP@IoU(%) | mAP@IoU(%)   |
|--------------|--------------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|
|              |              |                |         0.1  |         0.2  |         0.3  |         0.4  |         0.5  | AVG          |
| Random       | -            | RGB(VIT)       |         1.82 |         0.92 |         0.48 |         0.23 |         0.09 | 0.71         |
| DeepMIL [12] | CVPR 2018    | RGB(ViT)       |        22.72 |        15.57 |         9.98 |         6.2  |         3.78 | 11.65        |
| AVVD [22]    | TMM 2022     | RGB(ViT)       |        30.51 |        25.75 |        20.18 |        14.83 |         9.79 | 20.21        |
| VadCLIP [21] | AAAI 2024    | RGB(ViT)       |        37.03 |        30.84 |        23.38 |        17.9  |        14.31 | 24.70        |
| AVadCLIP∗    | this work    | RGB(ViT)       |        39.63 |        32.77 |        26.84 |        21.58 |        16.39 | 27.44        |
| AVadCLIP     | this work    | RGB(ViT)+Audio |        41.89 |        34.61 |        27.08 |        22.16 |        17.3  | 28.61        |

TABLE IV FINE -GRAINED COMPARISONS ON CCTV-FIGHTSsub .

| Method       | Reference    | Modality       |   mAP@IoU(%) |   mAP@IoU(%) |   mAP@IoU(%) |   mAP@IoU(%) |   mAP@IoU(%) | mAP@IoU(%)   |
|--------------|--------------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|
|              |              |                |         0.1  |         0.2  |         0.3  |         0.4  |         0.5  | AVG          |
| VadCLIP [21] | AAAI 2024    | RGB(ViT)       |        19.34 |        14.32 |         9.25 |         6.64 |         3.73 | 10.66        |
| AVadCLIP∗    | this work    | RGB(ViT)       |        21.1  |        14.57 |         9.01 |         5.74 |         4.69 | 11.02        |
| AVadCLIP     | this work    | RGB(ViT)+Audio |        22.25 |        15.91 |        10.4  |         7    |         5.28 | 12.17        |

TABLE V CROSS -DATASET WSVAD RESULTS ON XD-VIOLENCE AND CCTV-FIGHTSsub .

| Test⇒ 
 Train⇓    | XD-Violence    |   CCTV-Fightssub
 AP(%) |
|-------------------|----------------|-------------------------|
| Train⇓            | AP(%)          |                   69.24 |
| XD-Violence       | 86.04          |                   69.24 |
| CCTV-Fightssub    | 76.60          |                   73.38 |

multimodal baselines, achieving a remarkable 4.9% gain over the latest method AVCL [49].

For fine-grained anomaly detection, AVadCLIP consistently outperforms all competitors across different IoU thresholds, as detailed in Table III. With RGB-only input, AVadCLIP ∗ surpasses VadCLIP at all IoU thresholds, achieving an AVG improvement of 2.7%. Similarly, the full-modality model AVadCLIP leads across all metrics, boosting the AVG by 3.9%. These results highlight the effectiveness of multimodal learning in precisely localizing anomaly boundaries and improving category predictions.

2) Performance comparison on CCTV-Fightssub: The coarse-grained anomaly detection results on CCTV-Fightssub are presented in Table II. For RGB-only methods, AVadCLIP ∗ achieves 73.36% AP, surpassing the state-of-the-art VadCLIP and demonstrating the effectiveness of our approach in unimodal scenarios. For audio-visual scenarios, AVadCLIP further improves performance, outperforming all existing methods. These results indicate that incorporating audio information can further enhance anomaly detection performance, validating the effectiveness of cross-modal complementary information mining.

We present the fine-grained anomaly detection results on CCTV-Fightssub in Table IV, and it can be observed that AVadCLIP consistently outperforms all competitors at different IoU thresholds. Using RGB-only input, AVadCLIP ∗ surpasses VadCLIP at all IoU thresholds, achieving a 0.4% improvement in AVG. Similarly, AVadCLIP with audio leads in all metrics, increasing AVG by 1.5%. These results further highlight the effectiveness of multimodal learning in accurately locating

TABLE VI EFFECTIVENESS OF DESIGNED MODULES ON XD-VIOLENCE .

| AV    | V Fusion    | AV Prompt    | LF OCAL    |   AP(%)  |   AVG(%) |
|-------|-------------|--------------|------------|----------|----------|
|       | ×           | ×            | ×          |    79.85 |    27.89 |
|       | √           | ×            | ×          |    82.9  |    26.63 |
|       | √ 
 √       | √
 √         | ×          |    86.18 |    26.79 |
|       | √           | √            | √          |    86.04 |    28.61 |

anomalous boundaries and improving category prediction.

3) Cross-dataset Results: Table V presents the cross-dataset evaluation results of AVadCLIP on XD-Violence and CCTVFightssub, aiming to assess its generalization capability across different domains. Despite being trained on one dataset and tested on another, AVadCLIP consistently achieves competitive performance, demonstrating strong robustness and transferability. For example, AVadCLIP trained on XD-Violence still achieves an AP of 69.24% when directly tested on the surveillance-oriented CCTV-Fightssub, with less than a 4% drop compared to the model trained specifically on that dataset. These results highlight the model's ability to generalize well to unseen data distributions and diverse anomaly scenarios.

Overall, AVadCLIP achieves state-of-the-art performance in both unimodal and multimodal settings across coarse-grained and fine-grained anomaly detection tasks. The comprehensive results validate its effectiveness in leveraging audio-visual collaboration and demonstrate the feasibility of uncertaintydriven distillation strategy.

## D. Ablation Studies

1) The effect of audio-visual adaptive fusion: From Table VI, it can be observed that the introduction of audiovisual fusion improves detection performance. Furthermore, Table VII presents the impact of different audio-visual fusion strategies on anomaly detection performance. First, the cross attention fusion performs poorly in the WSVAD task, indicating that although it can capture the relationships between modalities, its complex parameterized design may negatively

Fig. 4. Coarse-grained and Fine-grained WSVAD visualization results of AVadCLIP and the baseline model on XD-Violence.

![Image](artifacts/image_000003_4f209ded85235536a62fc68d0ab97bd74e33dab0e0aafc4cfb72206cc221e3e3.png)

TABLE VII EFFECTIVENESS OF AUDIO -VISUAL FUSION ON XD-VIOLENCE .

| Method                   |   AP(%)  |   AVG(%) |
|--------------------------|----------|----------|
| Cross Attention          |    75.15 |    10.51 |
| Element-wise Addition    |    83.02 |    27.66 |
| Concat+Linear Projection |    83.36 |    28.88 |
| Adaptive Fusion          |    86.04 |    28.61 |

TABLE VIII EFFECTIVENESS OF UKD ON XD-VIOLENCE .

| Method              |   AP(%)  |   AVG(%) |
|---------------------|----------|----------|
| Audio Model w/o UKD |    50.89 |    12.2  |
| Audio Model w       |    52.51 |    13.5  |
| Visual Model w/o UK |    84.6  |    22.92 |
| Visual Model w      |    85.53 |    27.44 |
| Audio-Visual Model  |    86.04 |    28.61 |

TABLE IX EFFECTIVENESS OF UKD ON CCTV-FIGHTSsub .

| Method               |   AP(%)  |   AVG(%) |
|----------------------|----------|----------|
| Visual Model w/o UKD |    67.89 |    10.65 |
| Visual Model w       |    73.36 |    11.02 |
| Audio-Visual Model   |    73.38 |    12.17 |

impact the generalization ability of CLIP model in downstream WSVAD tasks. Next, the simple element-wise addition strategy achieves an AP of 83.02% and an AVG of 27.66%. Then, the concatenation with linear projection approach improves the AP to 83.36% and the AVG to 28.88%, indicating that enhancing feature representation through linear transformation facilitates more effective cross-modal information capture. Finally, our proposed adaptive fusion strategy achieves the best AP of 86.04%, outperforming the other three methods on the whole. This demonstrates that our adaptive fusion strategy, as a lightweight and effective fusion strategy, can more exploit complementary information between audio and

## visual modalities.

2) The effect of audio-visual prompt and LF OCAL: As presented in Table VI, the baseline model achieves an AP of only 79.85%. Integrating the audio-visual prompt on top of the adaptive fusion mechanism significantly enhances performance, increasing the AP to 86.18%. This improvement underscores the effectiveness of the audio-visual prompt in capturing critical multimodal patterns, thereby facilitating more precise anomaly recognition. Furthermore, incorporating focal loss into the model contributes to refining anomaly boundary detection, leading to more stable performance in fine-grained anomaly localization. In summary, the audio-visual prompt primarily enhances coarse-grained anomaly detection, and focal loss further refines boundary precision, enabling the model to achieve optimal performance across both AP and AVG metrics.

3) The effect of uncertainty-driven distillation: As shown in Table VIII, the proposed UKD mechanism significantly enhances anomaly detection performance in both visual-only and audio-only models. Specifically, in the visual-only setting, UKD achieves a 0.9% improvement in AP and a 4.5% increase in AVG, attaining performance levels comparable to the teacher model trained with audio-visual inputs. Similarly, the audio-only model also benefits from UKD, exhibiting consistent performance gains. These results highlight the effectiveness of UKD in leveraging data uncertainty to enhance the robustness of unimodal representations during the distillation process, making it particularly well-suited for real-world applications where modality incompleteness is prevalent. In addition, we present a comparison of the effectiveness of the UKD module on CCTV-Fightssub in Table IX. The results show that the proposed UKD mechanism significantly improves the anomaly detection performance of the unimodal model. Notably, in the visual-only scenario, adding UKD improves AP by 5.5%, achieving performance comparable to that of the audio-visual model. This finding further demonstrates the effectiveness of UKD in enhancing the robustness of unimodal representations through data uncertainty.

Fig. 5. Coarse-grained WSVAD visualization results of AVadCLIP, AVadCLIP ∗ , and the baseline model on XD-Violence.

![Image](artifacts/image_000004_608312dd9cb25588aec612ab1ace3a184897c6a52032f9649a835e9c6a48e32c.png)

Fig. 6. Coarse-grained WSVAD visualization results of AVadCLIP, AVadCLIP ∗ , and the baseline model on CCTV-Fightssub .

![Image](artifacts/image_000005_3b0d3fca2f97c4ebbf4c23c5033ed47df79639773592d397212140cd27d16dd0.png)

## E. Qualitative Results

In Figure 4, we present the qualitative visualizations of AVadCLIP and the baseline model for both coarse-grained and fine-grained WSVAD. The blue curves denote the anomaly predictions by AVadCLIP, whereas the yellow curves represent those by the baseline model (RGB-only w/o UKD). As illustrated, compared to the baseline model, AVadCLIP significantly reduces anomaly confidence in normal video segments, thereby enhancing its ability to distinguish between abnormal and normal regions more accurately. The fine-grained map below also indicates that AVadCLIP can predict categories with greater precision. Notably, the observed performance improvement supports our hypothesis that audio information is more advantageous in visual occlusion (shooting) or acoustic dominant scenes (explosion), and can effectively eliminate ambiguity in visually similar patterns in anomaly detection scenes, thereby ensuring more robust detection performance.

In addition, we compare the coarse-grained visualization results of the baseline model (RGB-only w/o UKD), the student model AVadCLIP ∗ , and the teacher model AVadCLIP on XD-Violence, as shown in Figure 5. Experimental results show that AVadCLIP significantly outperforms the other two counterparts. By using this model as a teacher model to guide the unimodal student model, it effectively mitigates anomaly confidence biases, steering them towards more accurate detection results. To a certain extent, this demonstrates the robustness of our proposed method.

In order to demonstrate the superiority of our proposed method more comprehensively and intuitively, Figure 6 shows the coarse-grained visualization results on the CCTV-Fightssub dataset (since this dataset only includes the "Fighting" category, fine-grained visualizations are not provided). It can be seen that our method achieves significantly higher anomaly confidence scores in abnormal regions and notably lower scores in normal regions compared to the baseline model. This demonstrates that the integration of audio and video information can still yield substantial performance improvements in complex scenes. Besides, as can be seen from the last two rows, the unimodal model distilled with the UKD mechanism shows significantly fewer false positives compared to the baseline, demonstrating that the UKD mechanism effectively transfers audio-visual multi-modal knowledge into the unimodal model.

## V. CONCLUSION

In this work, we propose a novel weakly supervised framework for robust video anomaly detection using audio-visual collaboration. Leveraging the powerful representation ability

and cross-modal alignment capability of CLIP, we design two distinct modules to achieve efficient audio-visual collaboration and multimodal anomaly detection, based on the frozen CLIP model. Specifically, to seamlessly integrate audio-visual information, we introduce a lightweight fusion mechanism that adaptively generates fusion weights based on the importance of audio to assist visual information. Additionally, we propose an audio-visual prompt strategy that dynamically refines text embeddings with key multimodal features, strengthening the semantic alignment between video content and corresponding textual labels. To further bolster robustness in scenarios with missing modalities, we develop an uncertainty-driven distillation module that synthesizes audio-visual representations from visual inputs, focusing on challenging features. Experimental results across two benchmarks demonstrate that our framework effectively enables video-audio anomaly detection and enhances the model's robustness in scenarios with incomplete modalities. In the future, we will explore the integration of additional modalities (e.g., textual description) based on VLMs to achieve more robust video anomaly detection.

## REFERENCES

- [1] W. Luo, W. Liu, D. Lian, and S. Gao, "Future frame prediction network for video anomaly detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7505–7520, 2021.
- [2] H. Lv, C. Zhou, Z. Cui, C. Xu, Y. Li, and J. Yang, "Localizing anomalies from weakly-labeled videos," IEEE transactions on image processing , vol. 30, pp. 4505–4515, 2021.
- [3] P. Wu and J. Liu, "Learning causal temporal relation and feature discrimination for anomaly detection," IEEE Transactions on Image Processing, vol. 30, pp. 3513–3527, 2021.
- [4] M. I. Georgescu, R. T. Ionescu, F. S. Khan, M. Popescu, and M. Shah, "A background-agnostic framework with adversarial training for abnormal event detection in video," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 9, pp. 4505–4523, 2021.
- [5] M. Z. Zaheer, J.-H. Lee, A. Mahmood, M. Astrid, and S.-I. Lee, "Stabilizing adversarially learned one-class novelty detection using pseudo anomalies," IEEE Transactions on Image Processing, vol. 31, pp. 5963– 5975, 2022.
- [6] C. Cao, Y. Lu, and Y. Zhang, "Context recovery and knowledge retrieval: A novel two-stream framework for video anomaly detection," IEEE Transactions on Image Processing, 2024.
- [7] T. Liu, K.-M. Lam, and B.-K. Bao, "Injecting text clues for improving anomalous event detection from weakly labeled videos," IEEE Transactions on Image Processing, 2024.
- [8] Y. Pu, X. Wu, L. Yang, and S. Wang, "Learning prompt-enhanced context features for weakly-supervised video anomaly detection," IEEE Transactions on Image Processing, 2024.
- [9] P. Wu, C. Pan, Y. Yan, G. Pang, P. Wang, and Y. Zhang, "Deep learning for video anomaly detection: A review," arXiv preprint arXiv:2409.05383, 2024.
- [10] P. Wu, J. Liu, X. He, Y. Peng, P. Wang, and Y. Zhang, "Toward video anomaly retrieval from video anomaly detection: New benchmarks and model," IEEE Transactions on Image Processing, vol. 33, pp. 2213– 2225, 2024.
- [11] Y. Liu, H. Wang, Z. Wang, X. Zhu, J. Liu, P. Sun, R. Tang, J. Du, V. C. Leung, and L. Song, "Crcl: Causal representation consistency learning for anomaly detection in surveillance videos," IEEE Transactions on Image Processing, 2025.
- [12] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6479–6488.
- [13] M. Z. Zaheer, A. Mahmood, M. Astrid, and S.-I. Lee, "Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16. Springer, 2020, pp. 358–376.
- [14] C. Huang, C. Liu, J. Wen, L. Wu, Y. Xu, Q. Jiang, and Y. Wang, "Weakly supervised video anomaly detection via self-guided temporal discriminative transformer," IEEE Transactions on Cybernetics, 2022.
- [15] Z. Yang, J. Liu, and P. Wu, "Text prompt with normality guidance for weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 18 899–18 908.
- [16] J.-X. Zhong, N. Li, W. Kong, S. Liu, T. H. Li, and G. Li, "Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 1237–1246.
- [17] S. Paul, S. Roy, and A. K. Roy-Chowdhury, "W-talc: Weakly-supervised temporal activity localization and classification," in Proceedings of the European conference on computer vision, 2018, pp. 563–579.
- [18] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [19] C. Ju, T. Han, K. Zheng, Y. Zhang, and W. Xie, "Prompting visuallanguage models for efficient video understanding," in Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXV. Springer, 2022, pp. 105–124.
- [20] H. Xu, G. Ghosh, P.-Y. Huang, D. Okhonko, A. Aghajanyan, F. Metze, L. Zettlemoyer, and C. Feichtenhofer, "Videoclip: Contrastive pre-training for zero-shot video-text understanding," arXiv preprint arXiv:2109.14084, 2021.
- [21] P. Wu, X. Zhou, G. Pang, L. Zhou, Q. Yan, P. Wang, and Y. Zhang, "Vadclip: Adapting vision-language models for weakly supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 6, 2024, pp. 6074–6082.
- [22] P. Wu, X. Liu, and J. Liu, "Weakly supervised audio-visual violence detection," IEEE Transactions on Multimedia, pp. 1674–1685, 2022.
- [23] Y. Tian, J. Shi, B. Li, Z. Duan, and C. Xu, "Audio-visual event localization in unconstrained videos," in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 247–263.
- [24] Y. Tian, D. Li, and C. Xu, "Unified multisensory perception: Weaklysupervised audio-visual video parsing," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer, 2020, pp. 436–454.
- [25] P. Wu, J. Liu, Y. Shi, Y. Sun, F. Shao, Z. Wu, and Z. Yang, "Not only look, but also listen: Learning multimodal violence detection under weak supervision," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16. Springer, 2020, pp. 322–339.
- [26] H. Zhou, J. Yu, and W. Yang, "Dual memory units with uncertainty regulation for weakly supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, 2023.
- [27] J. Carreira and A. Zisserman, "Quo vadis, action recognition? a new model and the kinetics dataset," in proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.
- [28] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3d convolutional networks," in Proceedings of the IEEE international conference on computer vision, 2015, pp. 4489–4497.
- [29] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, "Audio set: An ontology and humanlabeled dataset for audio events," in 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2017, pp. 776–780.
- [30] C. Huang, J. Wen, Y. Xu, Q. Jiang, J. Yang, Y. Wang, and D. Zhang, "Self-supervised attentive generative adversarial networks for video anomaly detection," IEEE transactions on neural networks and learning systems, vol. 34, no. 11, pp. 9389–9403, 2022.
- [31] C. Shi, C. Sun, Y. Wu, and Y. Jia, "Video anomaly detection via sequentially learning multiple pretext tasks," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10 330–10 340.
- [32] C. Huang, J. Wen, C. Liu, and Y. Liu, "Long short-term dynamic prototype alignment learning for video anomaly detection," in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence , 2024, pp. 866–874.
- [33] Y. Cong, J. Yuan, and J. Liu, "Sparse reconstruction cost for abnormal event detection," in CVPR 2011. IEEE, 2011, pp. 3449–3456.
- [34] W. Luo, W. Liu, and S. Gao, "A revisit of sparse coding based anomaly detection in stacked rnn framework," in Proceedings of the IEEE international conference on computer vision, 2017, pp. 341–349.
- [35] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly detection–a new baseline," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6536– 6545.

- [36] C. Cao, H. Zhang, Y. Lu, P. Wang, and Y. Zhang, "Scene-dependent prediction in latent space for video anomaly detection and anticipation," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [37] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, "Learning temporal regularity in video sequences," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 733–742.
- [38] D. Gong, L. Liu, V. Le, B. Saha, M. R. Mansour, S. Venkatesh, and A. v. d. Hengel, "Memorizing normality to detect anomaly: Memoryaugmented deep autoencoder for unsupervised anomaly detection," in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 1705–1714.
- [39] Z. Yang, J. Liu, Z. Wu, P. Wu, and X. Liu, "Video event restoration based on keyframes for video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 14 592–14 601.
- [40] S. Li, F. Liu, and L. Jiao, "Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 2, 2022, pp. 1395–1403.
- [41] J.-C. Feng, F.-T. Hong, and W.-S. Zheng, "Mist: Multiple instance selftraining framework for video anomaly detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14 009–14 018.
- [42] M. Cho, M. Kim, S. Hwang, C. Park, K. Lee, and S. Lee, "Look around for anomalies: Weakly-supervised anomaly detection via context-motion relational learning," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 12 137–12 146.
- [43] F.-L. Chen, D.-Z. Zhang, M.-L. Han, X.-Y. Chen, J. Shi, S. Xu, and B. Xu, "Vlp: A survey on vision-language pre-training," Machine Intelligence Research, vol. 20, no. 1, pp. 38–56, 2023.
- [44] L. Zanella, W. Menapace, M. Mancini, Y. Wang, and E. Ricci, "Harnessing large language models for training-free video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 527–18 536.
- [45] Y. Yang, K. Lee, B. Dariush, Y. Cao, and S.-Y. Lo, "Follow the rules: Reasoning for video anomaly detection with large language models," arXiv preprint arXiv:2407.10299, 2024.
- [46] P. Wu, X. Zhou, G. Pang, Z. Yang, Q. Yan, P. WANG, and Y. Zhang, "Weakly supervised video anomaly detection and localization with spatio-temporal prompts," in ACM Multimedia 2024, 2024.
- [47] P. Wu, X. Zhou, G. Pang, Y. Sun, J. Liu, P. Wang, and Y. Zhang, "Openvocabulary video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 297–18 307.
- [48] D. Ding, L. Wang, L. Zhu, T. Gedeon, and P. Koniusz, "Learnable expansion of graph operators for multi-modal feature fusion," arXiv preprint arXiv:2410.01506, 2025.
- [49] J. Meng, H. Tian, G. Lin, J.-F. Hu, and W.-S. Zheng, "Audio-visual collaborative learning for weakly supervised video anomaly detection," IEEE Transactions on Multimedia, 2025.
- [50] J. Leng, Z. Wu, M. Tan, Y. Liu, J. Gan, H. Chen, and X. Gao, "Beyond euclidean: Dual-space representation learning for weakly supervised video violence detection," in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 17 373–17 397. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2024/file/1f471322127d6347e5ae09a14b1e5cf7-Paper-Conference.pdf
- [51] G. Li, Y. Wei, Y. Tian, C. Xu, J.-R. Wen, and D. Hu, "Learning to answer questions in dynamic audio-visual scenarios," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 19 108–19 118.
- [52] Y. Wei, D. Hu, Y. Tian, and X. Li, "Learning in audio-visual context: A review, analysis, and new perspective," arXiv preprint arXiv:2208.09579 , 2022.
- [53] Y. Chen, Y. Liu, H. Wang, F. Liu, C. Wang, H. Frazer, and G. Carneiro, "Unraveling instance associations: A closer look for audio-visual segmentation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 497–26 507.
- [54] J. Ma, P. Sun, Y. Wang, and D. Hu, "Stepping stones: a progressive training strategy for audio-visual semantic segmentation," in European Conference on Computer Vision. Springer, 2024, pp. 311–327.
- [55] R. Guo, L. Qu, D. Niu, Y. Qi, W. Yue, J. Shi, B. Xing, and X. Ying, "Open-vocabulary audio-visual semantic segmentation," in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 7533–7541.
- [56] X. He, X. Liu, Y. Li, D. Zhao, G. Shen, Q. Kong, X. Yang, and Y. Zeng, "Cace-net: Co-guidance attention and contrastive enhancement for effective audio-visual event localization," in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 985–993.
- [57] H. Xu, R. Zeng, Q. Wu, M. Tan, and C. Gan, "Cross-modal relationaware networks for audio-visual event localization," in Proceedings of the 28th ACM international conference on multimedia, 2020, pp. 3893– 3901.
- [58] J. Zhou, D. Guo, Y. Zhong, and M. Wang, "Advancing weaklysupervised audio-visual video parsing via segment-wise pseudo labeling," International Journal of Computer Vision, vol. 132, no. 11, pp. 5308–5329, 2024.
- [59] J. Zhou, D. Guo, Y. Mao, Y. Zhong, X. Chang, and M. Wang, "Labelanticipated event disentanglement for audio-visual video parsing," in European Conference on Computer Vision. Springer, 2024, pp. 35–51.
- [60] J. Chalk, J. Huh, E. Kazakos, A. Zisserman, and D. Damen, "Tim: A time interval machine for audio-visual action recognition," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 153–18 163.
- [61] Y. Liu, Z. Wu, M. Mo, J. Gan, J. Leng, and X. Gao, "Dual space embedding learning for weakly supervised audio-visual violence detection," in 2024 IEEE International Conference on Multimedia and Expo (ICME) . IEEE, 2024, pp. 1–6.
- [62] J. Yu, J. Liu, Y. Cheng, R. Feng, and Y. Zhang, "Modality-aware contrastive instance learning with self-distillation for weakly-supervised audio-visual violence detection," in Proceedings of the 30th ACM international conference on multimedia, 2022, pp. 6278–6287.
- [63] W.-F. Pang, Q.-H. He, Y.-j. Hu, and Y.-X. Li, "Violence detection in videos based on fusing visual and audio information," in ICASSP 20212021 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2021, pp. 2260–2264.
- [64] Z. Tong, Y. Song, J. Wang, and L. Wang, "Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training," Advances in neural information processing systems, vol. 35, pp. 10 078– 10 093, 2022.
- [65] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, "Scaling up visual and vision-language representation learning with noisy text supervision," in International Conference on Machine Learning. PMLR, 2021, pp. 4904–4916.
- [66] B. Ni, H. Peng, M. Chen, S. Zhang, G. Meng, J. Fu, S. Xiang, and H. Ling, "Expanding language-image pretrained models for general video recognition," in Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IV. Springer, 2022, pp. 1–18.
- [67] J. Cho, J. Lei, H. Tan, and M. Bansal, "Unifying vision-and-language tasks via text generation," in International Conference on Machine Learning. PMLR, 2021, pp. 1931–1942.
- [68] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, "Videochat: Chat-centric video understanding," arXiv preprint arXiv:2305.06355, 2023.
- [69] H.-H. Wu, P. Seetharaman, K. Kumar, and J. P. Bello, "Wav2clip: Learning robust audio representations from clip," in ICASSP 20222022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 4563–4567.
- [70] C. Lea, R. Vidal, A. Reiter, and G. D. Hager, "Temporal convolutional networks: A unified approach to action segmentation," in Computer vision–ECCV 2016 workshops: Amsterdam, the Netherlands, October 8-10 and 15-16, 2016, proceedings, part III 14. Springer, 2016, pp. 47–54.
- [71] X. Chen, S. Fischer, M. C. Rue, A. Zhang, D. Mukherjee, P. O. Kanold, J. Gillis, and A. M. Zador, "Whole-cortex in situ sequencing reveals input-dependent area identity," Nature, pp. 1–10, 2024.
- [72] X. Chen, N. Mishra, M. Rohaninejad, and P. Abbeel, "Pixelsnail: An improved autoregressive generative model," in International conference on machine learning. PMLR, 2018, pp. 864–872.
- [73] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, "Learning to prompt for visionlanguage models," International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.
- [74] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss ´ ´ for dense object detection," in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.
- [75] J. Chang, Z. Lan, C. Cheng, and Y. Wei, "Data uncertainty learning in face recognition," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5710–5719.
- [76] Z. Yang, W. Dong, X. Li, J. Wu, L. Li, and G. Shi, "Self-feature distillation with uncertainty modeling for degraded image recognition,"

- in European Conference on Computer Vision. Springer, 2022, pp. 552– 569.
- [77] M. Perez, A. C. Kot, and A. Rocha, "Detection of real-world fights in surveillance videos," in ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019, pp. 2662–2666.
- [78] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro, "Weakly-supervised video anomaly detection with robust temporal feature magnitude learning," in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 4975–4986.
- [79] H. K. Joo, K. Vo, K. Yamazaki, and N. Le, "Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection," in 2023 IEEE International Conference on Image Processing (ICIP) . IEEE, 2023, pp. 3230–3234.
- [80] L. Zanella, B. Liberatori, W. Menapace, F. Poiesi, Y. Wang, and E. Ricci, "Delving into clip latent space for video anomaly recognition," Computer Vision and Image Understanding, vol. 249, p. 104163, 2024.
- [81] C. Zhang, G. Li, Y. Qi, S. Wang, L. Qing, Q. Huang, and M.-H. Yang, "Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16 271–16 280.