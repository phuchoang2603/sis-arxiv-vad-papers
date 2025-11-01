---
title: 'VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning'
type: method
categories:
- Hybrid
github_link: https://github.com/GVCLab/VAU-R1
description: Introduces VAU-R1, a reinforcement fine-tuning framework leveraging
  Group Relative Policy Optimization (GRPO) to enhance multimodal large language
  models' (MLLMs) reasoning capabilities in video anomaly understanding (VAU). 
  Develops VAUBench, a comprehensive Chain-of-Thought benchmark with rich 
  annotations across perception, grounding, reasoning, and classification tasks,
  supported by multiple evaluation metrics including VAU-Eval, QA accuracy, 
  temporal IoU, and Factual Consistency. Demonstrates significant improvements 
  over supervised fine-tuning in question answering accuracy, temporal 
  localization, and interpretability, thereby establishing a scalable, 
  interpretable, and reasoning-aware VAU framework.
benchmarks:
- ucf-crime
- shanghaitech
- other
authors:
- Liyun Zhu
- Qixiang Chen
- Xi Shen
- Xiaodong Cun
date: '2023-10-01'
---

## VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning

Liyun Zhu 1 , 2 , ∗ Qixiang Chen 1 Xi Shen 3 Xiaodong Cun 2 , †

1 Australian National University 2 GVC Lab, Great Bay University 3 Intellindust AI Lab

{liyun.zhu, u7227010}@anu.edu.au, shenxiluc@gmail.com, cun@gbu.edu.cn

## Abstract

Video Anomaly Understanding (VAU) is essential for applications such as smart cities, security surveillance, and disaster alert systems, yet remains challenging due to its demand for fine-grained spatio-temporal perception and robust reasoning under ambiguity. Despite advances in anomaly detection, existing methods often lack interpretability and struggle to capture the causal and contextual aspects of abnormal events. This limitation is further compounded by the absence of comprehensive benchmarks for evaluating reasoning ability in anomaly scenarios. To address both challenges, we introduce VAU-R1, a data-efficient framework built upon Multimodal Large Language Models (MLLMs), which enhances anomaly reasoning through Reinforcement Fine-Tuning (RFT). Besides, we propose VAUBench, the first Chain-of-Thought benchmark tailored for video anomaly reasoning, featuring multiple-choice QA, detailed rationales, temporal annotations, and descriptive captions. Empirical results show that VAU-R1 significantly improves question answering accuracy, temporal grounding, and reasoning coherence across diverse contexts. Together, our method and benchmark establish a strong foundation for interpretable and reasoning-aware video anomaly understanding. Our code is available at https://github.com/GVCLab/VAU-R1 .

Figure 1: Effectiveness of Reinforcement Fine-Tuning. We compare QA accuracy and temporal anomaly grounding performance across different models. VAU-R1, trained via Reinforcement Fine-Tuning (RFT), consistently outperforms its Supervised Fine-Tuning (SFT) counterpart. This demonstrates that RFT enhances both reasoning and temporal localization capabilities in VAU tasks.

![Image](artifacts/image_000000_b00c208704cb5acb1a71feb703e229a7c67277c3416d419fcf814e59ca78eea5.png)

## 1 Introduction

Anomalies are events or behaviors that deviate from regular patterns or expected activities in a given context. In surveillance settings, these may include incidents such as fighting, theft, or

∗ Work done while the author was a visiting student at GVC Lab, Great Bay University.

† Corresponding Author

Figure 2: Overview of VAU-R1. VAU-R1 leverages Reinforcement Fine-Tuning to enhance the reasoning ability of MLLMs for video anomaly understanding. Specifically, we adopt Group Relative Policy Optimization (GRPO) to optimize the model with task-specific rewards, such as answer format, accuracy, and temporal Intersection-over-Union (IoU). We decompose the VAU task into four complementary tasks to facilitate comprehensive reasoning: multiple-choice QA, temporal anomaly grounding, anomaly reasoning, and anomaly classification.

![Image](artifacts/image_000001_2542ea5f1df8f57d1377dce0335c86b90cf4322d134fcd56ddc6fe07cbf88330.png)

traffic violations, etc. Video Anomaly Understanding (VAU) aims to detect and interpret such irregular events in unstructured, real-world video streams [22]. The task is challenging due to scene complexity, context dependence, varying camera viewpoints, and diverse anomaly types [27 , 44 , 56]. Early approaches only focuses on detecting anomalies, which typically framed the task as binary classification, assigning normal or abnormal labels to individual frames and identifying the temporal boundaries of anomalous events [5 , 6 , 11 , 16 , 21 , 33 , 36 , 47 , 55]. While effective for localization, these methods offer limited interpretability and provide little insight into the underlying causes of anomalies [7 , 8 , 49]. Recent advances in Multi-modal Large Language Models (MLLMs) have introduced the ability to generate textual descriptions of anomalous events [9 , 48 , 51 , 52 , 53], improving model transparency to some extent. However, current methods still face three key limitations: (i) they lack the ability to generate coherent, multi-step reasoning chains; (ii) no comprehensive benchmark provides rich annotations to support detailed causal reasoning; and (iii) evaluation protocols for reasoning quality remain underdeveloped.

To move beyond shallow classification and toward deeper understanding, we decompose VAU into four progressive stages: (i) Perception — identifying the scene and relevant objects, either through free-text descriptions or guided multiple-choice questions; (ii) Grounding — localizing the precise temporal segment where the anomaly occurs; (iii) Reasoning — explaining the event by analyzing causal factors, temporal dynamics, and contextual cues; and (iv) Conclusion — summarizing the event with a final decision, such as assigning it to a specific category (e.g., fighting vs. robbery). This structured formulation enables models to progressively build semantic understanding and supports more interpretable and task-aligned evaluation.

To implement this four-stage formulation, we introduce VAU-R1, a Reinforcement Fine-Tuning (RFT) framework designed to improve the reasoning capabilities of MLLMs on the VAU task. Our method builds on Group Relative Policy Optimization (GRPO) [31], incorporating task-specific reward signals based on answer format correctness, question-answer accuracy, and temporal grounding alignment. The framework is data-efficient and can be applied in low-resource settings, making it practical for real-world deployments. To support training and evaluation, we also construct VAUBench, a new benchmark that spans diverse scenarios and provides rich annotations across the four reasoning stages, including multiple-choice QA pairs, detailed event descriptions, temporal groundings, and step-by-step rationales. Finally, we propose a set of evaluation metrics—QA

accuracy, temporal Intersection-over-Union (IoU), GPT-based reasoning score, and classification accuracy—to quantitatively assess model performance across perception, grounding, reasoning, and conclusion. Together, VAU-R1 and VAU-Bench offer a scalable and unified framework for advancing structured video anomaly understanding. Our contribution can be summarized as follows:

- We propose VAU-R1, a data-efficient Reinforcement Fine-Tuning framework that improves the reasoning ability of MLLMs for video anomaly understanding. It outperforms standard supervised fine-tuning on reasoning-intensive tasks.
- We present VAU-Bench, the first large-scale benchmark with Chain-of-Thought annotations designed for video anomaly reasoning. It contains a diverse collection of videos, QA pairs, temporal labels, and detailed rationales spanning a wide range of real-world scenarios.
- We design a unified evaluation protocol that measures model performance across four reasoning stages, jointly considering reasoning quality, answer correctness, and temporal localization to capture both interpretability and detection precision.

## 2 Related Works

From Detection to Understanding. Early efforts in Video Anomaly Detection (VAD) can be broadly categorized into self-supervised and weakly-supervised paradigms. Self-supervised methods rely solely on normal video samples, learning the distribution of normal behavior and flagging deviations as anomalies [11 , 21 , 25]. In contrast, weakly-supervised methods are trained with both normal and anomalous videos using coarse video-level labels rather than fine-grained frame-level annotations [5 , 16 , 33 , 36 , 45 , 47 , 55]. These approaches typically adopt a top-k selection strategy to identify the most likely anomalous segments. While effective for localizing anomaly boundaries, they often rely heavily on motion cues [56], operating under the assumption that rapid or irregular motion is indicative of anomalous behavior. However, this assumption does not hold for subtle or semantically complex anomalies, leading to poor interpretability. To address these limitations, recent work has turned to video anomaly understanding, leveraging MLLMs to provide more semantically grounded and interpretable reasoning [26].

Prompt-Based vs. Learning-Based Approaches for VAU. Building on the shift toward semantic understanding, recent approaches to VAU fall into two main categories: prompt-based and learningbased methods. Prompt-based methods typically use MLLMs as anomaly scorers [30 , 51], or as reasoning agents via rule-based few-shot prompting [48] or learned question templates [49]. While these methods avoid computationally expensive training, their generalization ability is often limited due to the absence of task-specific adaptation. On the other hand, pretraining [8] and finetuning [52 , 53] approaches aim to learn anomaly-aware representations by incorporating video captions and causal reasoning signals (e.g., cause and effect). Despite this progress, existing methods remain constrained to improving anomaly description and fail to capture the full logical chain of an anomaly. To overcome these limitations, we leverage reinforcement fine-tuning to enhance the model's reasoning ability, enabling end-to-end identification of both when and why anomalies occur.

Reinforcement Learning in MLLMs. With the rise of powerful models such as OpenAI-o1 [15] and DeepSeek-R1 [12], reinforcement learning has been increasingly adopted in the post-training stage of MLLMs to enhance their reasoning capabilities [3 , 10 , 14 , 42 , 54]. While effective, this process often demands substantial computational resources and large-scale datasets, making it less practical for targeted downstream tasks [34]. To address these challenges, Visual-RFT [23] introduces Reinforcement Fine-Tuning (RFT) for visual tasks, demonstrating improved data efficiency and stronger performance compared to Supervised Fine-Tuning (SFT). Building on this idea, VideoChatR1 [17] extends RFT to video domains, achieving promising results in tasks such as question answering, temporal grounding, and object tracking. Yet, these tasks remain fragmented and have not been unified under the video anomaly understanding setting. To bridge this gap, we propose a framework that jointly addresses multiple tasks, aiming to advance comprehensive and interpretable anomaly reasoning.

## 3 Methodology

## 3.1 Preliminary: Reinforcement Learning via Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) [31] is a reinforcement learning framework that optimizes a policy πθ using preference-based feedback and multi-aspect reward signals. Given a question x, GRPO generates M candidate outputs O = {o1, o2, . . . , oM} from the old policy πθ old , each output oj assigned a reward Rj computed as a weighted sum of K task-specific components:

<!-- formula-not-decoded -->

where R (k) j is the k-th task-specific reward (e.g., accuracy, IoU, format compliance) and λk is its weight. To measure the relative quality of the j-th output, we calculate the normalized reward R ˜ j for each output oj with the mean µR and standard deviation σR across M candidates:

<!-- formula-not-decoded -->

GRPO maximises the following objective while keeping the update close to the original MLLM parameters πref through a KL penalty term DKL(· || ·):

<!-- formula-not-decoded -->

where β is a regularization coefficient. This formulation allows GRPO to incorporate diverse reward signals while retaining training stability through KL regularization.

## 3.2 VAU-R1

As shown in Figure 2, VAU-R1 is a data-efficient reinforcement fine-tuning framework designed for the four VAU tasks, including Multi-choice QA, Temporal Anomaly Grounding, Anomaly Reasoning, and Anomaly Classification. Given videos and task-specific questions, we fine-tune a pre-trained MLLM to improve its multi-step reasoning ability across different tasks. The model generates multiple candidate responses for each input, which are then scored using task-specific reward functions (e.g., accuracy, temporal IoU, or format compliance). We employ Group Relative Policy Optimization (GRPO) to optimize the model, which maximizes reward-weighted likelihood while constraining divergence from the reference model via KL regularization. Our reinforcement-based approach outperforms supervised fine-tuning (SFT) in both reasoning capability and generalization to unseen scenarios. The design of task-specific reward functions is further detailed in Section 3.3 .

## 3.3 Reward Rules

We adopt the general idea of GRPO-based RFT to optimize the VAU model by designing task-specific reward functions for different VAU components. Below, we detail each reward definition.

Format Reward. For multiple-choice QA and anomaly classification tasks, we instruct the model to enclose its reasoning within &lt;think&gt;...&lt;/think&gt; tags and the answer within &lt;answer&gt;...&lt;/answer&gt; tags. For the temporal anomaly grounding task, we additionally require &lt;glue&gt;...&lt;/glue&gt; tags to indicate the predicted time span in seconds. The reward is defined as:

<!-- formula-not-decoded -->

We apply a format reward to VAU tasks to enforce structured outputs and discourage format violations.

Accuracy Reward. We also define an accuracy reward R acc to measure the correctness of the model's answer. In our experiments, this reward is given by:

<!-- formula-not-decoded -->

Figure 3: Statistics of our VAU-Bench. (a) Distribution of main anomaly types. (b) Distribution of video durations (top) and the proportion of anomalous segments within each video (bottom). (c) The evaluation criteria for four VAU tasks.

![Image](artifacts/image_000002_ec45de56454cb0aff1939e5296e47771920cfc88932523a32f165d2437eec726.png)

This simple accuracy reward encourages the model to choose the right answer during training.

Temporal IoU Reward. To encourage precise temporal grounding, we introduce a temporal Intersection-over-Union (IoU) reward RtIoU, which measures the alignment between the predicted and ground truth anomaly intervals. The reward is defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(6)

Here, [s1, s2] denotes the predicted temporal span of the anomaly, while [s ∗ ∗
1
, s ∗ 2 ] is the ground truth interval. The temporal IoU quantifies the degree of overlap between these intervals, and serves as a fine-grained reward signal to guide the model toward more accurate temporal localization.

Task-specific Reward Formulations. We apply task-specific combinations of the reward components mentioned above. For the multiple-choice QA task, we use a combination of format and accuracy rewards: R QA = R format + R acc . For temporal anomaly grounding, we further include a temporal IoU term to evaluate localization quality: RTAG = Rformat + R acc + RtIoU. For anomaly classification, we adopt a similar reward design as QA: RCLS = Rformat + R acc.

## 3.4 VAU-Bench

Task Definition. We decompose the VAU task into four stages: perception, grounding, reasoning, and conclusion. These stages address four core questions respectively: "What happens in this video?", " When does the anomaly occur?", "Why does the anomaly happen?", and "What is our overall judgment of the anomaly?". Corresponding to these stages, we define four VAU tasks:

- Multiple-Choice QA: Targets event perception by answering questions about videos.
- Temporal Grounding: Localizes anomalous segments in the video timeline.
- Anomaly Reasoning: Explores causal relationships to explain why an anomaly arises.

Table 1: Comparison of performance on MSAD and UCF-Crime datasets on multiple-choice QA task and anomaly reasoning task. Accw/o think and Accw/ think refer to the multiple-choice question accuracy without and with thinking, respectively. For the anomaly reasoning task, CLS , KM , FLU , INF, and FAC represent VAU-Eval scores generated by DeepSeek-V3, measuring classification accuracy, key concept alignment, linguistic fluency, informativeness, and factual consistency, respectively. Each dimension is scored on a 10-point scale. Total denotes the aggregated score over five dimensions.

| Dataset    | Model               | QA Accuracy    | QA Accuracy    | VAU-Eval
 KM↑ FLU↑ INF↑ FAC↑ Total↑   | VAU-Eval
 KM↑ FLU↑ INF↑ FAC↑ Total↑   | VAU-Eval
 KM↑ FLU↑ INF↑ FAC↑ Total↑   | VAU-Eval
 KM↑ FLU↑ INF↑ FAC↑ Total↑   | VAU-Eval
 KM↑ FLU↑ INF↑ FAC↑ Total↑   | VAU-Eval
 KM↑ FLU↑ INF↑ FAC↑ Total↑   |
|------------|---------------------|----------------|----------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| Dataset    | Model               | Accw/o think   | Accw/ think    | CLS↑                                  | KM↑                                   | FLU↑                                  | INF↑                                  | FAC↑                                  | Total↑                                |
| MSAD       | InternVL2.5-2B      | 76.67          | 72.08          | 6.84                                  | 6.23                                  | 8.55                                  | 6.64                                  | 6.64                                  | 34.90                                 |
| MSAD       | Qwen2.5-VL-7B       | 84.58          | 83.33          | 6.75                                  | 6.41                                  | 9.27                                  | 7.74                                  | 6.92                                  | 37.08                                 |
| MSAD       | InternVL2.5-8B-MPO  | 82.50          | 84.17          | 6.83                                  | 6.33                                  | 8.32                                  | 6.37                                  | 6.86                                  | 34.72                                 |
| MSAD       | Qwen2-VL-2B         | 77.08          | 72.50          | 5.94                                  | 5.43                                  | 8.77                                  | 6.29                                  | 5.90                                  | 32.25                                 |
| MSAD       | +SFT                | 82.92          | 85.83          | 6.04                                  | 5.43                                  | 8.89                                  | 6.55                                  | 5.93                                  | 32.84                                 |
| MSAD       | +RFT                | 82.92 (↑5.84   | 83.75 (↑11.25) | 6.05(↑)                               | 5.49(↑)                               | 8.89(↑)                               | 6.50(↑)                               | 6.05(↑)                               | 32.98(↑)                              |
| MSAD       | Qwen2.5-VL-3B       | 85.83          | 82.50          | 5.77                                  | 5.24                                  | 9.02                                  | 6.74                                  | (↑
 570                               | 32.47                                 |
| MSAD       | +SFT                | 86.25          | 84.58          | 2.89                                  | 2.22                                  | 9.02 
 489                            | 3.52                                  | 2.44                                  | 15.96                                 |
| MSAD       | +RFT                | 88.33          | 87.08 (↑4.     | 2.89 
 597(↑                          | 2.22                                  | 4.89                                  | 6.84(↑                                | 2.44 
 603(                           | 15.96
 3338(                          |
| MSAD       | +RFT                | 88.33 (↑       | 87.08 (↑4.58)  | 5.97(↑)                               | 5.49(↑)                               | 9.05(↑)                               | 6.84(↑)                               | 6.03(↑)                               | 33.38(↑)                              |
|            | InternVL2.5-2B      | 84.86          | 68.13          | 4.40                                  | 3.08                                  | 8.09                                  | 5.69                                  | 3.47                                  | 24.74                                 |
|            | Qwen2.5-VL-7B       | 92.03          | 89.64          | 4.80                                  | 3.73                                  | 8.95                                  | 7.05                                  | 4.25                                  | 28.78                                 |
|            | InternVL 2.5 8B-MPO | 89.64          | 90.44          | 3.79                                  | 3.20                                  | 8.23                                  | 5.77                                  | 3.48                                  | 24.47                                 |
|            | Qwen2-VL-2B         | 87.25          | 83.67          | 3.47                                  | 2.48                                  | 7.75                                  | 4.49                                  | 2.82                                  | 21.02                                 |
|            | +SFT                | 83.67          | 86.06          | 3.61                                  | 2.26                                  | 7.30                                  | 4.79                                  | 2.70                                  | 20.66                                 |
|            | +RFT                | 88.45 (↑1.20)  | 88.05 (↑4.38)  | 4.04(↑)                               | 2.75(↑)                               | 7.72(↓)                               | 4.89(↑)                               | 3.11(↑)                               | 22.52(↑)                              |
|            | Qwen2.5-VL-3B       | 91.63          | 83.27          | 4.31                                  | 2.88                                  | 8.70                                  | 5.95                                  | 3.27                                  | 25.10                                 |
|            | +SFT 
 +RFT         | 92.03 (↑0.40)  | 91.63 (↑8.36)  | 1.80                                  | 1.01                                  | 4.15                                  | 2.82                                  | 1.11                                  | 10.89                                 |
|            | +RFT                | 92.03 (↑0.40)  | 91.63 (↑8.36)  | 4.42(↑)                               | 2.98(↑)                               | 8.71(↑)                               | 5.98(↑)                               | 3.39(↑)                               | 25.49(↑)                              |

- Anomaly Classification: Assigns the anomaly to its corresponding category.

This structured decomposition provides a clear framework for systematically addressing different perspectives of VAU, with each task rigorously evaluated using domain-specific metrics.

Dataset Construction and Annotation. Existing video anomaly datasets typically provide only frame-level labels [1 , 33 , 56] or sparse descriptions [8 , 9 , 50], limiting their usefulness for reasoningbased tasks. To address this, we construct VAU-Bench, a unified benchmark built from MSAD [56], UCF-Crime [33], and ECVA [8], enriched with Chain-of-Thought (CoT) annotations, including: (i) video descriptions, (ii) temporal boundaries, (iii) multiple-choice QA, and (iv) reasoning rationales. We apply a cleaning pipeline to remove corrupted or overly long videos and merge overlapping anomaly types. For UCF-Crime and ECVA, we use DeepSeek-V3 [18] to generate video-level summaries, QA pairs, and reasoning chains. For MSAD, CoT annotations are produced through a two-stage pipeline: we first apply InternVL-8B-MPO [42] to generate initial captions and analyses, which are then verified and refined using DeepSeek-V3 to obtain more accurate QA pairs and coherent reasoning rationales. We also give further construction and annotation details in the Appendix.

Dataset Statistics. Figure 3 presents an overview of VAU-Bench, the first VAU benchmark designed for Chain-of-Thought reasoning. Our dataset contains 4,602 videos covering 19 major anomaly types, with a total duration of 169.1 hours. It includes over 1.5 million words of fine-grained textual annotations, averaging 337 words per video, encompassing detailed descriptions, reasoning rationales, and multiple-choice questions. The dataset is split into 2,939 training, 734 validation, and 929 test videos. Additionally, we provide 3,700 temporal annotations to support the anomaly grounding task. Figure 3a shows the distribution of the main anomaly categories, while Figure 3b illustrates the diversity in video duration and anomaly sparsity. The evaluation protocols and metrics used for different tasks are summarized in Figure 3c, and we give more dataset statistics in the Appendix.

Reasoning Evaluation Metric: VAU-Eval. For VAU tasks, prior work has adopted BLEU and ROUGE [8 , 35 , 53] to evaluate semantic content. However, such n-gram-based metrics often fall short in capturing reasoning quality and deeper relational understanding. To better assess anomaly reasoning, we propose VAU-Eval, a GPT-based metric that compares model-generated descriptions and analyses with ground truth annotations. As illustrated in Figure 3c, we evaluate each response along five dimensions using DeepSeek-V3 [18] as the judge: classification accuracy, key concept

Table 2: Comparison of temporal anomaly grounding performance on the three datasets. For each dataset, we present results for the base models, followed by SFT and RFT variants. w/o think and w/ think refer to the inference prompt without and with thinking, respectively. Rows highlighted in light yellow denote the results on the UCF-Crime dataset, serving as an out-of-distribution test for cross-dataset evaluation.

| Dataset    | Model         | w/o think      | w/o think    | w/o think    | w/o think    | w/ think
 @03 @0@0   | w/ think
 @03 @0@0   | w/ think
 @03 @0@0   | w/ think
 @03 @0@0   |
|------------|---------------|----------------|--------------|--------------|--------------|----------------------|----------------------|----------------------|----------------------|
| Dataset    | Model         | mIoU           | R@0.3        | R@0.5        | R@0.7        | mIoU                 | R@0.3                | R@0.5                | R@0.7                |
| MSAD       | Qwen2-VL-2B   | 0.00           | 0.00         | 0.00         | 0.00         | 0.00                 | 0.00                 | 0.00                 | 0.00                 |
| MSAD       | Qwen2.5-VL-7B | 45.90          | 70.83        | 45.83        | 21.67        | 17.57                | 26.67                | 11.67                | 3.33                 |
| MSAD       | Qwen2.5-VL-3B | 21.27          | 30.00        | 10.83        | 4.17         | 13.00                | 16.67                | 5.83                 | 1.67                 |
| MSAD       | + SFT         | 30.65          | 47.50        | 30.00        | 9.17         | 35.17                | 50.83                | 34.17                | 15.00                |
| MSAD       | + RFT         | 35.77 (↑14.50) | 53.33        | 34.17        | 15.83        | 30.70 (↑17.7         | 48.33                | 29.17                | 12.50                |
| ECVA       | Qwen2-VL-2B   | 0.00           | 0.00         | 0.00         | 0.00         | 0.17                 | 0.30                 | 0.00                 | 0.00                 |
| ECVA       | Qwen2.5-VL-7B | 19.85          | 25.87        | 15.17        | 9.70         | 5.71                 | 7.96                 | 4.73                 | 2.99                 |
| ECVA       | Qwen2.5-VL-3B | 14.21          | 17.16        | 6.47         | 3.23         | 6.35                 | 7.21                 | 1.99                 | 0.50                 |
| ECVA       | + SFT         | 45.30          | 66.67        | 49.75        | 24.13        | 45.96                | 65.67                | 51.00                | 26.12                |
| ECVA       | + RFT         | 35.09 (↑20.88) | 49.00        | 28.86        | 19.40        | 33.25 (↑26.90)       | 48.51                | 30.60                | 18.41                |
| UCF-Crime  | Qwen2-VL-2B   | 2.74           | 4.84         | 0.00         | 0.00         | 0.12                 | 0.00                 | 0.00                 | 0.00                 |
| UCF-Crime  | Qwen2.5-VL-7B | 22.72          | 33.87        | 16.13        | 8.06         | 4.89                 | 8.06                 | 1.61                 | 0.00                 |
| UCF-Crime  | Qwen2.5-VL-3B | 10.91          | 15.32        | 6.45         | 3.23         | 7.68                 | 10.48                | 4.84                 | 1.61                 |
| UCF-Crime  | + SFT         | 4.98           | 3.23         | 0.81         | 0.00         | 5.76                 | 5.65                 | 0.81                 | 0.81                 |
| UCF-Crime  | + RFT         | 16.80 (↑5.89)  | 23.39        | 8.06         | 4.03         | 9.21 (↑1.53)         | 9.68                 | 4.03                 | 1.61                 |

alignment, fluency, informativeness, and factual consistency. Each dimension is scored on a 10-point scale to provide fine-grained assessment of reasoning quality.

## 4 Experiment

Implementation Details. Our main experiments are conducted using the Qwen2-VL-2B-Instruct [41] and Qwen2.5-VL-3B-Instruct [2] models. We apply full-parameter fine-tuning without adapters or LoRA, using 2 NVIDIA H20 GPUs for training. During the RFT training process, we adopt a structured prompting strategy that guides the model to generate intermediate reasoning and final answers in a standardized format. Specifically, each prompt instructs the model to enclose its reasoning process within &lt;think&gt;...&lt;/think&gt; tags and its final answer within &lt;answer&gt;...&lt;/answer&gt; tags. This format ensures consistency across different tasks. During inference, for Qwen-VL models, we sample frames at 1 FPS. For InternVL models, we uniformly sample 16 frames per video.

## 4.1 Evaluation of VAU-R1

Evaluation Protocol. We report results separately on the MSAD, ECVA, and UCF-Crime datasets rather than using a single aggregated benchmark, as these datasets differ substantially in anomaly types, video durations, and scene contexts. All evaluation metrics for our four tasks are summarized in Figure 3c. For the QA task, we report multiple-choice accuracy. Temporal anomaly grounding is evaluated using temporal mean Intersection over Union (mIoU), as well as recall at different IoU thresholds: R@0.3, R@0.5, and R@0.7. For anomaly reasoning, we adopt the GPT-based VAU-Eval introduced in Section 3.4. Finally, binary and multi-class classification accuracy are used for evaluating the anomaly classification task.

Evaluation on QA-Guided Reasoning. As shown in Table 1, we evaluate the reasoning capabilities of VAU-R1 on MSAD and UCF-Crime using multiple-choice QA accuracy and GPT-based VAU-Eval scores. We highlight two key observations. First, base models often perform worse when generating answers with reasoning (Accw/think) compared to without (Accw/o think) reasoning, indicating that naive Chain-of-Thought generation may introduce hallucination. In contrast, reinforcement fine-tuning (RFT) improves both QA accuracy with reasoning (e.g., +11.25 on MSAD) and overall reasoning quality. Second, RFT leads to consistent gains across five VAU-Eval dimensions—classification, demonstrating its ability to strengthen structured reasoning. For instance, on MSAD, Qwen2.5-VL3B+RFT achieves the highest total VAU-Eval score (33.38), showing substantial improvement over

Table 3: Ablation study on task co-training for anomaly classification. Bin. Acc. denotes binary classification accuracy (normal vs. abnormal), and Multi Acc. denotes multi-class accuracy over 19 anomaly types plus the normal class. Results are reported with and without think prompting.

| Model                             | w/o think    | w/o think    | w/ think   | w/ think   |
|-----------------------------------|--------------|--------------|------------|------------|
| Model                             | Bin. Acc.    | Multi Acc.   | Bin. Acc   | Multi Acc. |
| Baseline (Qwen2.5-VL-3B-Instruct) | 62.77        | 47.96        | 59.33      | 39.06      |
| +SFT w/ CLS                       | 81.12        | 29.08        | 83.37      | 32.19      |
| +RFT w/ CLS                       | 60.30        | 46.14        | 59.01      | 42.27      |
| +RFT w/ QA                        | 59.01        | 46.14        | 58.91      | 41.95      |
| +RFT w/ TAG                       | 67.81        | 49.46        | 74.14      | 46.14      |
| +RFT w/ QA-TAG                    | 65.77        | 47.53        | 67.60      | 45.06      |
| +RFT w/ QA-TAG-CLS                | 64.70        | 48.61        | 65.02      | 45.60      |

its SFT counterpart. These results confirm that RFT not only enhances answer correctness but also fosters robust and generalizable multimodal reasoning under the VAU setting.

Evaluation on Temporal Anomaly Grounding. As shown in Table 2, we evaluate the temporal anomaly grounding performance across three datasets. Note that all models are trained only on MSAD and ECVA, while UCF-Crime serves as an out of distribution test set. We observe several key findings. First, across both inference settings (w/ and w/o think), RFT consistently outperforms the corresponding base models, demonstrating its effectiveness in improving temporal localization. Notably, the RFT-finetuned 3B model achieves higher mIoU than the larger 7B base model on ECVA. Second, similar to our observations in QA-guided reasoning, Chain-of-Thought prompting does not necessarily enhance grounding performance. In some cases, adding reasoning leads to degraded localization accuracy. Third, RFT shows significantly better generalization compared to SFT. In cross-dataset evaluation (e.g., UCF-Crime as an out-of-distribution test), SFT demonstrates limited generalization, whereas RFT maintains strong performance across unseen scenarios. While SFT occasionally outperforms RFT in isolated cases, we observe that its direct predictions are opaque and lack interpretability, often yielding repetitive, non-discriminative outputs across videos (see Figure 4). These results highlight the advantages of RFT for enhancing generalization in VAU tasks.

Ablation Study. For VAU, the core objective is to make accurate high-level judgments about anomaly categories (e.g., distinguishing a fight from a robbery). To explore effective task formulations, we train models with different combinations of VAU tasks—multiple-choice QA, temporal anomaly grounding (TAG), and multi-class classification (CLS)—to assess their impact on reasoning. As shown in Table 3, RFT models trained with TAG alone achieve the highest binary accuracy (74.14) and strong multi-class performance (46.14) under the think setting, highlighting the benefit of temporal grounding for perception and category discrimination. Combining QA and TAG also improves performance but is slightly less effective than TAG alone. In contrast, SFT tends to over-predict anomalies, yielding high binary accuracy but poor multi-class results, suggesting overfitting. Overall, grounding-based tasks are more effective for anomaly classification, and jointly optimizing tasks via reinforcement learning yields complementary gains in both accuracy and reasoning.

Case Study. Figure 4 illustrates two representative examples from the QA and TAG tasks, comparing SFT and our VAU-R1 under the same Chain-of-Thought (CoT) prompt. In the QA example, SFT incorrectly selects a normal explanation based on surface cues, while VAU-R1 correctly infers a people-falling anomaly by identifying posture and behavioral irregularities. In the TAG example, SFT outputs a coarse anomaly span without rationale, whereas VAU-R1 localizes the anomaly more precisely (0.0–13.6s) and provides an interpretable causal chain. These cases highlight VAU-R1's superior reasoning and interpretability in both classification and localization settings. More qualitative case studies are provided in the Appendix.

## 4.2 Discussion

RFT Enhances Generalization and Interpretability. Our experiments demonstrate that RFT consistently outperforms SFT across multiple VAU tasks, offering improved interpretability (Table 1) and better generalization (Table 2). In contrast, SFT tends to memorize task-specific patterns and suffers from poor generalization to unseen scenarios. This suggests that SFT-trained models are more prone to overfitting, especially when trained on limited or narrowly defined tasks.

Is Chain-of-Thought Reasoning Necessary for VAU? Our findings suggest that Chain-of-Thought (CoT) reasoning does not always lead to better performance in visual understanding tasks. However,

Figure 4: Qualitative case of the QA (top) and TAG (bottom) task. All ground-truths and correct answers are highlighted in orange. Both SFT and RFT perform inference using the same CoT prompt. RFT's explicit chain-of-thought yields precise, interpretable QA choice and anomaly interval, whereas SFT's output is less informative and tends to produce inaccurate responses.

![Image](artifacts/image_000003_ac79e7a124ac332369b814df7e69f9ae31a6576a8240b4e4b62adad5a8ee0bd2.png)

it significantly enhances interpretability by providing structured justifications. Unlike mathematical or logical tasks, where reasoning is more deterministic, visual understanding involves inherently diverse reasoning paths. Therefore, designing simpler sub-tasks with well-defined reward signals to guide reasoning effectively remains underexplored. Directly applying complex tasks (e.g., multi-class anomaly classification) without task co-training often leads to suboptimal results (Table 3).

Rethinking Anomaly Understanding in Multimodal Contexts. VAU calls for constructing a coherent reasoning chain that bridges spatial-temporal localization and causal inference. Yet, leveraging diverse cues such as keyframes, salient objects, and even additional modalities (e.g., audio) to support unified reasoning remains underexplored. We envision that future work could benefit from integrating these multimodal signals into a structured reasoning framework, enabling more robust and interpretable anomaly understanding. Our method and benchmark take a step in this direction by proposing a unified evaluation protocol across perception, localization, and reasoning dimensions, ultimately guiding models toward accurate and justifiable anomaly judgments.

## 5 Conclusion

We present VAU-R1, an advanced and unified Video Anomaly Understanding framework focusing on four VAU tasks: multi-choice QA, temporal grounding, anomaly reasoning, and classification. VAU-R1 leverages a multimodal large language model (MLLM) and, notably, employs reinforcement fine-tuning to enhance anomaly reasoning and explainability via carefully designed GRPO reward functions for each task. To facilitate the training and evaluation of this framework, we also introduce VAU-Bench, the first chain-of-thought benchmark designed to train and evaluate VAU tasks at the

reasoning level. The experiments on different tasks prove the strong performance of the proposed method than baselines.

## References

- [1] A. Acsintoae, A. Florescu, M.-I. Georgescu, T. Mare, P. Sumedrea, R. T. Ionescu, F. S. Khan, and M. Shah. Ubnormal: New benchmark for supervised open-set video anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20143–20153, 2022.
- [2] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] J. Bi, S. Liang, X. Zhou, P. Liu, J. Guo, Y. Tang, L. Song, C. Huang, G. Sun, J. He, et al. Why reasoning matters? a survey of advancements in multimodal reasoning (v1). arXiv preprint arXiv:2504.03151, 2025.
- [4] C. Cao, Y. Lu, P. Wang, and Y. Zhang. A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20392–20401, 2023.
- [5] Y. Chen, Z. Liu, B. Zhang, W. Fok, X. Qi, and Y.-C. Wu. Mgfn: Magnitude-contrastive glanceand-focus network for weakly-supervised video anomaly detection. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 387–395, 2023.
- [6] D. Ding, L. Wang, L. Zhu, T. Gedeon, and P. Koniusz. Lego: Learnable expansion of graph operators for multi-modal feature fusion. arXiv preprint arXiv:2410.01506, 2024.
- [7] K. Doshi and Y. Yilmaz. Towards interpretable video anomaly detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2655–2664, 2023.
- [8] H. Du, G. Nan, J. Qian, W. Wu, W. Deng, H. Mu, Z. Chen, P. Mao, X. Tao, and J. Liu. Exploring what why and how: A multifaceted benchmark for causation understanding of video anomaly. arXiv preprint arXiv:2412.07183, 2024.
- [9] H. Du, S. Zhang, B. Xie, G. Nan, J. Zhang, J. Xu, H. Liu, S. Leng, J. Liu, H. Fan, et al. Uncovering what why and how: A comprehensive benchmark for causation understanding of video anomaly. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18793–18803, 2024.
- [10] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, B. Wang, and X. Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [11] D. Gong, L. Liu, V. Le, B. Saha, M. R. Mansour, S. Venkatesh, and A. v. d. Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1705–1714, 2019.
- [12] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [13] K. Hara, H. Kataoka, and Y. Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 6546–6555, 2018.
- [14] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Z. Xu, Y. Hu, and S. Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749 , 2025.
- [15] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [16] J. Leng, Z. Wu, M. Tan, Y. Liu, J. Gan, H. Chen, and X. Gao. Beyond euclidean: Dual-space representation learning for weakly supervised video violence detection. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [17] X. Li, Z. Yan, D. Meng, L. Dong, X. Zeng, Y. He, Y. Wang, Y. Qiao, Y. Wang, and L. Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.
- [18] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [19] K. Liu, W. Liu, C. Gan, M. Tan, and H. Ma. T-c3d: Temporal convolutional 3d network for real-time action recognition. In Proceedings of the AAAI conference on artificial intelligence , volume 32, 2018.
- [20] K. Liu and H. Ma. Exploring background-bias for anomaly detection in surveillance videos. In Proceedings of the 27th ACM International Conference on Multimedia, pages 1490–1499, 2019.
- [21] W. Liu, W. Luo, D. Lian, and S. Gao. Future frame prediction for anomaly detection–a new baseline. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 6536–6545, 2018.
- [22] Y. Liu, D. Yang, Y. Wang, J. Liu, J. Liu, A. Boukerche, P. Sun, and L. Song. Generalized video anomaly event detection: Systematic taxonomy and comparison of deep models. ACM Computing Surveys, 56(7):1–38, 2024.
- [23] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [24] C. Lu, J. Shi, and J. Jia. Abnormal event detection at 150 fps in matlab. In Proceedings of the IEEE international conference on computer vision, pages 2720–2727, 2013.
- [25] Y. Lu, F. Yu, M. K. K. Reddy, and Y. Wang. Few-shot scene-adaptive anomaly detection. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pages 125–141. Springer, 2020.
- [26] H. Lv and Q. Sun. Video anomaly detection and explanation via large language models. arXiv preprint arXiv:2401.05702, 2024.
- [27] G. Pang, C. Shen, L. Cao, and A. V. D. Hengel. Deep learning for anomaly detection: A review. ACM computing surveys (CSUR), 54(2):1–38, 2021.
- [28] B. Ramachandra and M. Jones. Street scene: A new dataset and evaluation protocol for video anomaly detection. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2569–2578, 2020.
- [29] R. Rodrigues, N. Bhargava, R. Velmurugan, and S. Chaudhuri. Multi-timescale trajectory prediction for abnormal human activity detection. In The IEEE Winter Conference on Applications of Computer Vision (WACV), March 2020.
- [30] Y. Shao, H. He, S. Li, S. Chen, X. Long, F. Zeng, Y. Fan, M. Zhang, Z. Yan, A. Ma, et al. Eventvad: Training-free event-aware video anomaly detection. arXiv preprint arXiv:2504.13092 , 2025.
- [31] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [32] K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems, 27, 2014.
- [33] W. Sultani, C. Chen, and M. Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6479–6488, 2018.

- [34] H. Tan, Y. Ji, X. Hao, M. Lin, P. Wang, Z. Wang, and S. Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.
- [35] J. Tang, H. Lu, R. Wu, X. Xu, K. Ma, C. Fang, B. Guo, J. Lu, Q. Chen, and Y. Chen. Hawk: Learning to understand open-world video anomalies. Advances in Neural Information Processing Systems, 37:139751–139785, 2024.
- [36] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4975–4986, 2021.
- [37] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 4489–4497, 2015.
- [38] M. Vijay, W.-X. LI, B. Viral, and V. Nuno. Anomaly detection in crowded scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1975–1981, 2010.
- [39] L. Wang, W. Li, W. Li, and L. Van Gool. Appearance-and-relation networks for video classification. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 1430–1439, 2018.
- [40] L. Wang, Y. Xiong, Z. Wang, Y. Qiao, D. Lin, X. Tang, and L. Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In European conference on computer vision, pages 20–36. Springer, 2016.
- [41] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [42] W. Wang, Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, J. Zhu, X. Zhu, L. Lu, Y. Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.
- [43] X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7794–7803, 2018.
- [44] P. Wu, C. Pan, Y. Yan, G. Pang, P. Wang, and Y. Zhang. Deep learning for video anomaly detection: A review. arXiv preprint arXiv:2409.05383, 2024.
- [45] P. Wu, X. Zhou, G. Pang, Y. Sun, J. Liu, P. Wang, and Y. Zhang. Open-vocabulary video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18297–18307, 2024.
- [46] P. Wu, X. Zhou, G. Pang, Z. Yang, Q. Yan, P. Wang, and Y. Zhang. Weakly supervised video anomaly detection and localization with spatio-temporal prompts. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 9301–9310, 2024.
- [47] P. Wu, X. Zhou, G. Pang, L. Zhou, Q. Yan, P. Wang, and Y. Zhang. Vadclip: Adapting visionlanguage models for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 6074–6082, 2024.
- [48] Y. Yang, K. Lee, B. Dariush, Y. Cao, and S.-Y. Lo. Follow the rules: reasoning for video anomaly detection with large language models. In European Conference on Computer Vision , pages 304–322. Springer, 2024.
- [49] M. Ye, W. Liu, and P. He. Vera: Explainable video anomaly detection via verbalized learning of vision-language models. arXiv preprint arXiv:2412.01095, 2024.
- [50] T. Yuan, X. Zhang, K. Liu, B. Liu, C. Chen, J. Jin, and Z. Jiao. Towards surveillance videoand-language understanding: New dataset baselines and challenges. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22052–22061, 2024.

- [51] L. Zanella, W. Menapace, M. Mancini, Y. Wang, and E. Ricci. Harnessing large language models for training-free video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18527–18536, 2024.
- [52] H. Zhang, X. Xu, X. Wang, J. Zuo, C. Han, X. Huang, C. Gao, Y. Wang, and N. Sang. Holmesvad: Towards unbiased and explainable video anomaly detection via multi-modal llm. arXiv preprint arXiv:2406.12235, 2024.
- [53] H. Zhang, X. Xu, X. Wang, J. Zuo, X. Huang, C. Gao, S. Zhang, L. Yu, and N. Sang. Holmesvau: Towards long-term video anomaly understanding at any granularity. arXiv preprint arXiv:2412.06171, 2024.
- [54] H. Zhou, X. Li, R. Wang, M. Cheng, T. Zhou, and C.-J. Hsieh. R1-zero's" aha moment" in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.
- [55] H. Zhou, J. Yu, and W. Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 3769–3777, 2023.
- [56] L. Zhu, L. Wang, A. Raj, T. Gedeon, and C. Chen. Advancing video anomaly detection: A concise review and a new dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

## A Further Dataset Details

Figure 5: More dataset statistics of our VAU-Bench. (a) Distribution of training, validation, and test splits across the four tasks included in VAU-Bench. (b) Word cloud visualization of frequent terms appearing in the multiple-choice questions and choices.

![Image](artifacts/image_000004_7ff91fbe07e5d749b92fcaef0f22f51c0864591063f0a002a3e69f02ab97b1c9.png)

Dataset Annotation. VAU-Bench is constructed from three datasets: UCF-Crime, ECVA, and MSAD. While UCF-Crime [33] and ECVA [8] provide basic scene-level descriptions, they lack the structured annotations necessary for fine-grained reasoning. To address this, we leverage DeepSeekV3 [18], a powerful large language model, to enrich the existing annotations from HIVAU-70K (which includes UCF-Crime) [53] and ECVA [8]. We use prompt-based instruction to guide the model in extracting key events, causal relationships, and anomalous behaviors, thereby producing reasoning-oriented annotations suitable for causal understanding. The detailed prompt design is provided in the blue-colored box below.

## Video Understanding Prompt.

```
You are an expert in video understanding and reasoning. I will give you structured metadata for a surveillance or behavior-related video. Your task is twofold: Please analyze the entire video description, including anomaly labels, events, and all textual summaries. Based on this, generate a comprehensive summary of what happens in the video in the following JSON structure:
```

```
{ "judgement": "Does this video depict an anomaly? If yes, what is it called?", "description": "Chronological and factual summary of what happened in the video", "analysis": { "Specific Anomaly Type": "Select from the [Anomaly Type]", "Location": "Where the event occurs: indoor/outdoor/specific", "Key Evidence": "Key actions or objects that support classification", "Detailed Explanation": "Why these events are normal/anomalous", "Cause and Effect": "What led to the event and its outcome", "Conclusion": "Wrap-up reasoning with final conclusion about the event" } }
```

## Generating QA Pair Prompt.

You are an expert in reasoning-focused QA generation for surveillance analysis videos. You will be given a structured video summary, including: (i) A judgement (whether the video is anomalous or normal). (ii) A chronological description of what happens in the video. (iii) A multi-part analysis that breaks down the event's anomaly type, location, key evidence, explanation, causes, and conclusion. Please generate a single multiple-choice question-answer pair in JSON format.

For the MSAD [56] dataset, which lacks textual annotations, we design a structured Chain-of-Thought (CoT) annotation pipeline. We first use InternVL2.5-8B-MPO [42] as the Vision-Language Model (VLM) to generate initial annotations that include detailed descriptions, step-by-step reasoning, and anomaly classification. To further improve the quality of these annotations, we apply DeepSeek-V3 in a secondary refinement stage, which enhances the coherence and clarity of the generated descriptions, QA pairs, and reasoning chains. The overall annotation pipeline consists of the following stages:

- Task Definition: The VLM is instructed to act as an anomaly detector.
- Video Description: The VLM generates a detailed description of the video content.
- Step-by-Step Reasoning: The VLM performs multi-step reasoning to analyze the presence and nature of anomalies.
- Verification: Given the ground-truth anomaly type, the VLM verifies whether its prediction aligns with it. If not, it regenerates both the description and reasoning.
- Key Object Summarization: The VLM identifies key visual objects or cues relevant to the anomaly, expressed in 1–3 words.
- QA Generation: The VLM constructs multiple-choice questions by generating and shuffling plausible anomaly-related answer options.
- Quality Enhancement: We use DeepSeek-V3 to validate and refine the generated QA pairs, descriptions, and reasoning chains.

After completing the CoT annotation for the entire VAU-Bench, we perform a manual review to ensure the accuracy and consistency of all generated annotations.

More Dataset Statistics. Table 4 presents a detailed comparison of our VAU-Bench and existing video anomaly datasets. Compared to previous datasets, our benchmark offers a longer total video duration, a more diverse set of primary anomaly types (with similar categories merged), diverse multi-choice QA pairs, and richer Chain-of-Thought reasoning annotations. Figure 5a shows the dataset splits across four tasks. Each task contains a balanced number of training, validation, and test samples, supporting robust evaluation. Figure 5b presents a word cloud of frequent phrases extracted from the multiple-choice questions and answers in VAU-Bench. Notably, the presence of phrases such as "best describes" , "plausible explanation", and "behavioral clue" highlights the variety of question formulations, encouraging models to engage in fine-grained interpretation. In addition, keywords such as robbery , man action, and scene indicate that our questions are intentionally crafted to guide models toward recognizing specific objects and anomaly types in complex real-world scenarios.

Dataset Examples. We present representative examples from our VAU-Bench, each annotated to support four core tasks of video anomaly understanding. As illustrated in Figure 7, each example is richly labeled with a question-answer pair, key visual evidence, anomaly type, temporal annotation, and a multi-part reasoning chain that includes location, cause and effect, and a high-level conclusion. This annotation format enables models not only to detect and classify anomalies, but also to explain them in a structured, interpretable manner. Figure 7 and Figure 8 show challenging anomaly scenarios, while Figure 9 depicts a normal scene, included to test model robustness and reduce false positives. These examples demonstrate the breadth and depth of our annotations, enabling holistic evaluation across perception and reasoning dimensions.

## B Experiment Details

Training Details. We use the Adam optimizer with a learning rate of 2 × 10 − 5 . The supervised fine-tuning (SFT) stage runs for less steps (e.g. 200) to avoid overfitting, while the Reinforcement Fine-Tuning (RFT) stage takes approximately 15 hours for 1.5k steps. We set the hyperparameter β in the KL divergence term of the GRPO to 0.04, using M = 4 candidate outputs per prompt. The maximum response length is capped at 1024 tokens.

Table 4: Comparison of video anomaly detection benchmarks. We compare VAU-Bench with existing datasets in terms of size, annotation granularity, and reasoning capabilities. VAU-Bench is the first benchmark to support structured reasoning via multiple-choice questions and Chain-of-Thought (CoT) annotations. Columns indicate whether each dataset provides QA pairs, free-text descriptions (Descrip.), anomaly judgement (Judge.), reasoning (Reason.), and full CoT rationales.

| Dataset               | Year #Videos Total Len. #Type    |   Year #Videos Total Len. #Type  | Year #Videos Total Len. #Type    | Year #Videos Total Len. #Type    | Annotation QA Pairs Descrip. Judge. Reason. CoT   | Annotation QA Pairs Descrip. Judge. Reason. CoT   | Annotation QA Pairs Descrip. Judge. Reason. CoT   | Annotation QA Pairs Descrip. Judge. Reason. CoT   | Annotation QA Pairs Descrip. Judge. Reason. CoT   | Annotation QA Pairs Descrip. Judge. Reason. CoT   |
|-----------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| UCSD Ped1 [38]        | 2010                             |                              70  | 0.1h                             | 5                                | Bounding-box                                      | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| UCSD Ped2 [38]        | 2010                             |                              28  | 0.1h                             | 5                                | Bounding-box                                      | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| CUHK Avenue [24]      | 2013                             |                              35  | 0.5h                             | 5                                | Bounding-box                                      | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| ShanghaiTech [21]     | 2017                             |                             437  | 3.5h                             | 13                               | Bounding-box                                      | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| UCF-Crime [33]        | 2018                             |                            1900  | 128.0h                           | 13                               | Frame                                             | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| Street Scene [28]     | 2020                             |                              81  | 3.8h                             | 17                               | Bounding-box                                      | %                                                 | %                                                 | %                                                 | % 
 %                                             | %                                                 |
| IITB Corridor [29]    | 2020                             |                             358  | 2.0h                             | 10                               | Frame                                             | %                                                 | %                                                 | % 
 %                                             | % 
 %                                             | %
 %                                              |
| UBNormal [1]          | 2022                             |                             543  | 2.2h                             | 22                               | Frame                                             | %                                                 | %                                                 | % 
 %                                             | %                                                 | %                                                 |
| NWPU [4]              | 2023                             |                             547  | 16.3h                            | 43                               | Frame                                             | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| MSAD [56]             | 2024                             |                             720  | 4.1h                             | 11                               | Frame                                             | %                                                 | %                                                 | %                                                 | %                                                 | %                                                 |
| UCA [50]              | 2024                             |                            1854  | 121.9h                           | 13 Time Duration                 | 13 Time Duration                                  | %                                                 | !                                                 | %                                                 | %                                                 | %                                                 |
| CUVA [9]              | 2024                             |                            1000  | 32.5h                            | 11 Time Duration                 | 11 Time Duration                                  | !                                                 | !                                                 | !                                                 | !                                                 | %                                                 |
| ECVA [8]              | 2024                             |                            2240  | 88.2h                            | 21 Time Duration                 | 21 Time Duration                                  | !                                                 | !                                                 | !                                                 | !                                                 | %                                                 |
| HIVAU-70K [53]        | 2025                             |                            5443  | NA                               | NA Time Duration                 | NA Time Duration                                  | !                                                 | !                                                 | %                                                 | %                                                 | %                                                 |
| VAU–Bench (Ours) 2025 | VAU–Bench (Ours) 2025            |                            4596  | 169.1h                           | 9 Time                           | 9 Time Duration                                   | !                                                 | !                                                 | !                                                 | !                                                 | !                                                 |

## VAU-Eval Prompt.

Below is a ground-truth description and analysis, followed by a model-generated description and analysis. Please evaluate the model's outputs from the following aspects:

1. Classification Correctness (10 pts)
2. Key Object and Action Matching (10 pts)
3. Fluency and Coherence (10 pts)
4. Informativeness and Domain Awareness (10 pts)
5. Factual Consistency (10 pts)

Evaluation Details for Anomaly Reasoning. To evaluate the alignment between model-generated outputs and our annotated ground truth in video anomaly understanding, we introduce VAU-Eval, a GPT-based evaluation protocol. The evaluation is structured as a multi-turn interaction, where the model first generates a description of the video and then performs reasoning to determine whether the video contains an anomaly. We then use DeepSeek-V3 [18] to assess the similarity between the predicted answers and the ground truth across five aspects: classification correctness, key object and action matching, fluency and coherence, informativeness and domain awareness, and factual consistency. Each aspect is scored out of 10 points, yielding a total of 50 points per sample. To better reflect the model's actual reasoning capabilities, we do not fine-tune the model on any reasoning-style description or analysis. Instead, we directly test models that are trained solely on the multiple-choice QA task, thus ensuring that their descriptive reasoning is not memorized but inferred. The detailed evaluation prompt used in this process is shown in the blue box above.

## C Further Evaluations

More Evaluations. As shown in Table 5, we conduct experiments on the ECVA dataset. Compared to UCF-Crime and MSAD, ECVA poses greater challenges across both recognition and reasoning tasks. All models consistently achieve lower VAU-Eval reasoning scores on ECVA, indicating that its longer videos, more camera movements, viewpoint shifts and richer anomaly diversity make fine-grained understanding more difficult. While our RFT-enhanced models achieve consistent improvements in multiple-choice QA accuracy, their VAU-Eval reasoning scores does not always improve. This suggests that while RFT helps models better predict the final answer, it does not necessarily enhance the reasoning process. These findings highlight the need for more fine-grained reward signals to guide the generation of high-quality rationales in complex scenarios.

Table 5: Comparison of performance on ECVA datasets on multiple-choice QA task and anomaly reasoning task. Accw/o think and Accw/ think refer to the multiple-choice question accuracy without and with thinking, respectively. For the anomaly reasoning task, CLS , KM , FLU , INF, and FAC represent VAU-Eval scores generated by DeepSeek-V3, measuring classification accuracy, key concept alignment, linguistic fluency, informativeness, and factual consistency, respectively. Each dimension is scored on a 10-point scale. Total denotes the aggregated score over five dimensions.

| Dataset    | Model              | QA Accuracy    | QA Accuracy    | VAU-Eval   | VAU-Eval   | VAU-Eval   | VAU-Eval   | VAU-Eval   | VAU-Eval   |
|------------|--------------------|----------------|----------------|------------|------------|------------|------------|------------|------------|
| Dataset    | Model              | Accw/o think   | Accw/ think    | CLS↑       | KM↑        | FLU↑       | INF↑       | FAC↑       | Total↑     |
|            | InternVL2.5-2B     | 78.84          | 58.84          | 2.86       | 2.78       | 7.57       | 4.62       | 3.03       | 20.86      |
|            | Qwen2.5-VL-7B      | 83.02          | 86.98          | 3.70       | 3.67       | 8.64       | 6.40       | 4.04       | 26.45      |
|            | InternVL2.5-8B-MPO | 90.00          | 83.72          | 3.4        | 3.31       | 7.87       | 4.48       | 3.47       | 22.53      |
|            | Qwen2-VL-2B        | 86.98          | 83.95          | 2.41       | 2.36       | 7.81       | 3.81       | 2.57       | 18.96      |
| ECVA       | +SFT               | 84.88          | 84.65          | 2.20       | 2.12       | 7.37       | 3.99       | 2.22       | 17.90      |
|            | +RFT               | 90.23 (↑3.25   | 84.42 (↑0.47   | 2.26       | 2.28       | 7.52       | 3.70       | 2.40       | 18.16      |
|            | Qwen2.5-VL-3B      | 85.58          | 75.81          | 2.21       | 2.58       | 8.33       | 5.02       | 2.75       | 20.89      |
|            | +SFT               | 89.30          | 86.98          | 1.50       | 1.22       | 4.37       | 2.66       | 1.24       | 10.99      |
|            | +RFT               | 89.53 (↑3.95   | 86.51 (↑10.7   | 1.45       | 2.24       | 8.05       | 4.32       | 2.39       | 18.45      |

Table 6: Performance of HolmesVAU 2B [53] and our VAU-R1 2B on multiple-choice QA and anomaly reasoning task.

| Model        | Dataset   | QA Accuracy    | QA Accuracy    | VAU-Eval
 ↑ ↑ AC↑ ↑   | VAU-Eval
 ↑ ↑ AC↑ ↑   | VAU-Eval
 ↑ ↑ AC↑ ↑   | VAU-Eval
 ↑ ↑ AC↑ ↑   | VAU-Eval
 ↑ ↑ AC↑ ↑   | VAU-Eval
 ↑ ↑ AC↑ ↑   | VAU-Eval
 ↑ ↑ AC↑ ↑   |
|--------------|-----------|----------------|----------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Model        | Dataset   | Accw/o think   | Accw/ think    | k                     | CLS↑                  | KM↑                   | FLU↑                  | INF↑                  | FAC↑                  | Total↑                |
| HolmesVAU 2B | MSAD      | 85.00          | 86.25          |                       | 2.7                   | 2.72                  | 6.82                  | 3.55                  | 3.33                  | 20.15                 |
| HolmesVAU 2B | UCF-Crime | 86.45          | 85.66          |                       | 3.05                  | 1.97                  | 6.30                  | 3.08                  | 2.39                  | 16.79                 |
| HolmesVAU 2B | ECVA      | 70.47          | 70.70          | 70                    | 2.54                  | 1.71                  | 6.26                  | 2.78                  | 2.30                  | 15.59                 |
| VAU-R1 2B    | MSAD      | 82.92          | 83.75          | 5 (↓2.5               | 6.05                  | 5.49                  | 8.89                  | 6.50                  | 6.05                  | 32.98                 |
| VAU-R1 2B    | UCF-Crime | 88.45 (↑       | 8.05 (↑2.      | 05 (↑2.39             | 4.04                  | 2.75                  | 7.72                  | 4.89                  | 3.11                  | 22.52                 |
| VAU-R1 2B    | ECVA      | 90.23 (↑19     | 84.42 (↑13.7   | 42 (↑13.72)           | 2.26                  | 2.28                  | 7.52                  | 3.70                  | 2.40                  | 18.16                 |

Comparison with Prior Work. As shown in Table 6, we evaluate HolmesVAU 2B [53], a recently released baseline for VAU, on our benchmark to assess its reasoning capability in complex scenarios. While HolmesVAU 2B achieves reasonable performance across all datasets, it consistently underperforms compared to our Qwen-based models, particularly on the challenging ECVA dataset. This performance gap is evident in both multiple-choice QA accuracy and VAU-Eval reasoning scores, indicating limitations in HolmesVAU 2B's ability to generalize to diverse and complex scenarios. In contrast, VAU-R1 demonstrates stronger alignment with human-annotated reasoning chains and greater robustness across datasets.

Classification Results. Table 7 presents the binary and multi-class anomaly classification accuracy on three datasets: MSAD, UCF-Crime, and ECVA. We directly apply the RFT strategy to train a multi-class anomaly classification task, which includes 19 different anomaly types as well as the normal class. However, directly training the complex multi-class task with RFT degrades performance, suggesting it is more effective to decompose the task into simpler sub-tasks with structured rewards to better guide learning. We compare multiple models under two settings: w/o think and w/ think . We observe that, for the relatively challenging multi-class anomaly task, incorporating an explicit "think" reasoning step improves the model's classification accuracy.

Temporal Localization Performance. Table 8 summarizes the temporal localization (mIoU) performance of representative methods, categorized into traditional models, multi-modal approaches, and MLLMs. As expected, early appearance-based methods (e.g., Two-stream [32], TSN [40], C3D [37]) achieve limited performance. Incorporating spatio-temporal modeling via 3D convolutions (T-C3D [19], ARTNet [39], 3DResNet [13]) brings moderate improvements, with Liu et al. [20] reaching a mIoU of 16.40. More recent multi-modal approaches, such as VADClip [47] and STPrompt [46], achieve significantly better performance, with STPrompt reaching 23.90 mIoU. Our MLLM-based methods show promising yet limited temporal grounding capabilities. While Qwen2.5-VL-3B achieves only 10.91 mIoU, reinforcement tuning (+RFT) boosts performance to 16.80, indicating that structured reward learning helps align model outputs with temporal structures.

Table 7: Comparison of anomaly classification accuracy on three datasets. Bin. Acc. denotes binary classification accuracy (normal vs. abnormal), and Multi Acc. denotes multi-class accuracy over 19 anomaly types and the normal class. Results are reported with and without think prompting.

| Dataset    | Model                                                                  | w/o think             | w/o think             | w/ think
 ccMulti Acc   | w/ think
 ccMulti Acc   |
|------------|------------------------------------------------------------------------|-----------------------|-----------------------|-------------------------|-------------------------|
|            |                                                                        | Bin. Acc.             | Multi Acc.            | Bin. Acc.               | Multi Acc.              |
| MSAD       | Qwen2-VL-2B-Instruct 
 Qwen2.5-VL-7B-Instruct 
 Qwen2.5-VL-3B-Instruct | 75.00 
 90.00 
 79.17 | 62.50 
 70.00 
 69.58 | 60.42 
 75.00 
 7333    | 52.92
 66.67            |
| MSAD       | Qwen2.5-VL-7B-Instruct                                                 | 90.00                 | 70.00                 | 75.00                   | 66.67                   |
| MSAD       | + SFT                                                                  | 70.83                 | 28.75                 | 74.58                   | 56.67
 3333             |
| MSAD       | + RFT                                                                  | 70.83 
 208 (↑29      | 71.25 (↑1.67)         | 74.58 
 74.58 (↑1.25    | 60.83 (↑4.16)           |
| MSAD       | + RFT                                                                  | 82.08 (↑2.91)         | 71.25 (↑1.67)         | 74.58 (↑1.25            | 60.83 (↑4.16)           |
| UCF-Crime  | Qwen2-VL-2B-Instruct 
 Qwen2.5-VL-7B-Instruct                          | 60.56 
 8685          | 53.78 
 6215          | 60.16                   | 51.79                   |
| UCF-Crime  | Qwen2.5-VL-7B-Instruct                                                 | 86.85                 | 62.15                 | 70.12                   | 61.35                   |
| UCF-Crime  | Qwen2.5-VL-3B-Instruct                                                 | 64.54                 | 58.57                 | 62.55                   | 52.19                   |
| UCF-Crime  | + SFT 
 + RFT                                                          | 64.14 
 62.55 (↓1.99  | 28.69 
 57.77 (↓0.80) | 69.32 
 62.15 (↓0.40    | 37.05                   |
| ECVA       | Qwen2-VL-2B-Instruct 
 Qwen25-VL-7B-Instruct                           | 41.95                 | 24.72 
 3288          | 32.88 
 4354            | 19.05
 2381             |
| ECVA       | Qwen2.5-VL-7B-Instruct                                                 | 64.85                 | 32.88                 | 43.54                   | 23.81                   |
| ECVA       | Qwen2.5-VL-3B-Instruct                                                 | 52.83                 | 30.16                 | 49.89                   | 22.00                   |
| ECVA       | + SFT                                                                  | 96.37                 | 29.48                 | 96.15                   | 28.80                   |
| ECVA       | + RFT                                                                  | 49.66 (↓3.17)         | 30.61 (↑0.45)         | 55.78 (↑5.89)           | 31.07 (↑9.07            |

Table 8: Comparison of temporal localization performance (mIoU) across different methods on UCF-Crime dataset.

| Category    | Method              | Feature    |   mIoU |
|-------------|---------------------|------------|--------|
| Multi-moda  | Two-stream [32]     | 2.20       |   2.2  |
| Multi-moda  | TSN [40]            | SN         |   2.6  |
| Multi-moda  | C3D [37]            | C3D        |   7.2  |
| Multi-moda  | T-C3D [19]          | C3D        |  10.2  |
| Multi-moda  | ARTNet [39]         | ARTNets    |  11.4  |
| Multi-moda  | 3DResNet [13]       | I3D-ResNe  |  10.3  |
| Multi-moda  | NLN [43]            | I3D-ResNe  |  12.2  |
| Multi-moda  | Liu et al. [20]     | I3D-ResNet |  16.4  |
| Multi-modal | VADClip [47]        | 22.05      |  22.05 |
| Multi-modal | p 
 STPrompt [46]   | CLIP       |  23.9  |
| MLLMs       | Qwen2.5-VL-3B       | 10.91      |  10.91 |
| MLLMs       | Qwen2.5-VL-3B + RFT | ViT        |  16.8  |
| MLLMs       | Qwen2.5-VL-7B       | ViT        |  22.72 |

However, even with RFT, MLLMs still underperform compared to specialized temporal models, suggesting that current architectures may lack explicit temporal reasoning modules required for fine-grained localization.

Case Study on Anomaly Reasoning. Figure 6 presents a qualitative comparison between outputs generated by SFT and our proposed VAU-R1 model on anomaly reasoning task. Both models are evaluated using the same Chain-of-Thought (CoT) prompt and scored based on five criteria: classification correctness (CLS), key object matching (KM), fluency (FLU), informativeness (INF), and factual consistency (FAC). The SFT output incorrectly identifies the anomaly as a political argument, which does not match the core issue (an escalator malfunction). It also fails to mention any key visual evidence or relevant location. In contrast, VAU-R1 produces a more contextually appropriate response, identifying an emergency situation at a subway station involving injured individuals and emergency vehicles. While the response focuses on surface-level emergency context rather than the root cause, it demonstrates greater fluency and relevance. The evaluation assigns a higher total score of 22, with solid performance across all dimensions, particularly in fluency and informativeness.

## D Limitation and Future Work

One limitation of this work is its focus on a constrained set of tasks, namely multiple-choice question answering, temporal grounding, anomaly reasoning, and anomaly classification. While these tasks form a strong foundation for video anomaly understanding, there remains substantial room for extension. Future work could incorporate additional tasks such as spatial localization of key objects, which would enable more fine-grained event understanding. Moreover, introducing additional modalities (e.g., audio) may provide complementary cues that enhance both the robustness and contextual depth of anomaly reasoning.

## E Potential Societal Impact

We propose a new method and benchmark for video anomaly understanding. Accurate and interpretable anomaly understanding systems can contribute to a wide range of safety-critical applications, such as disaster early warning, fire prevention, fall detection, and public safety monitoring. By enabling models to reason about abnormal events, our approach can assist first responders in identifying urgent situations earlier and more reliably.

However, this research inevitably involves scenarios that depict violent or chaotic abnormal behaviors. We strictly follow established ethical guidelines throughout our study. The datasets used in this study are publicly available and have been processed in accordance with the guidelines provided by their original publishers. We strictly adhere to these terms of use and employ the data solely for academic research purposes. To ensure privacy protection, the datasets include safeguards such as reduced video resolution and facial blurring, effectively preventing the identification of individuals. Looking ahead, we plan to explore anomaly understanding methods that incorporate privacy preservation as a core design principle.

Figure 6: Qualitative case of the Anomaly Reasoning task. All correct description and analysis are highlighted in orange. The evaluation results are presented on the right of the answer respectively. Both SFT and VAU-R1 perform inference using the same CoT prompt. VAU-R1's output correctly identifies the anomaly with high fluency but lacks reasoning for the core event, whereas SFT's output is inaccurate and tends to produce repetitive responses.

![Image](artifacts/image_000005_e7a2f284436da621883332c5742c1e6aa0c9a20566597b2cd03f4911647e194e.png)

![Image](artifacts/image_000006_eca921733253d817d650241f19bc50eadc6bbe8f4e0042fd6b5166a80767bcf5.png)

Figure 7: Example of VAU-Bench. An explosion case in an outdoor backyard, highlighting complex anomaly detection and dynamic scene understanding, labeled with a question-answer pair, key visual evidence, anomaly type, and a multi-part reasoning chain that includes location, cause and effect, and a high-level conclusion.

## Key Object: The perpetrators

Figure 8: Example of VAU-Bench. A stealing incident, demonstrating capabilities in human activity recognition and intent analysis.

![Image](artifacts/image_000007_0a8642bddd32605f9bec5c31531f3edfa6dea1b4bb4aba6d4bf754cad6c40666.png)

Figure 9: Example of VAU-Bench. A normal scene, used to evaluate model robustness against false positives and to enhance dataset diversity.

![Image](artifacts/image_000008_bdad07224af9b85427e7309135fb630e4508737fccd452fc9c95f41531b722b0.png)