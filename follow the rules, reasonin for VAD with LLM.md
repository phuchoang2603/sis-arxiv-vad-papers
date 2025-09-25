## Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models

Yuchen Yang 1⋆ , Kwonjoon Lee 2 , Behzad Dariush 2 , Yinzhi Cao 1 , and Shao-Yuan Lo 2

1

Johns Hopkins University {yc.yang, yinzhi.cao}@jhu.edu 2 Honda Research Institute USA {kwonjoon\_lee, bdariush, shao-yuan\_lo}@honda-ri.com

Abstract. Video Anomaly Detection (VAD) is crucial for applications such as security surveillance and autonomous driving. However, existing VAD methods provide little rationale behind detection, hindering public trust in real-world deployments. In this paper, we approach VAD with a reasoning framework. Although Large Language Models (LLMs) have shown revolutionary reasoning ability, we find that their direct use falls short of VAD. Specifically, the implicit knowledge pre-trained in LLMs focuses on general context and thus may not apply to every specific real-world VAD scenario, leading to inflexibility and inaccuracy. To address this, we propose AnomalyRuler, a novel rule-based reasoning framework for VAD with LLMs. AnomalyRuler comprises two main stages: induction and deduction. In the induction stage, the LLM is fed with few-shot normal reference samples and then summarizes these normal patterns to induce a set of rules for detecting anomalies. The deduction stage follows the induced rules to spot anomalous frames in test videos. Additionally, we design rule aggregation, perception smoothing, and robust reasoning strategies to further enhance AnomalyRuler's robustness. AnomalyRuler is the first reasoning approach for the one-class VAD task, which requires only few-normal-shot prompting without the need for full-shot training, thereby enabling fast adaption to various VAD scenarios. Comprehensive experiments across four VAD benchmarks demonstrate AnomalyRuler's state-of-the-art detection performance and reasoning ability. AnomalyRuler is open-source and available at: https://github.com/Yuchen413/AnomalyRuler

## 1 Introduction

Video Anomaly Detection (VAD) aims to identify anomalous activities, which are infrequent or unexpected in surveillance videos. It has a wide range of practical applications, including security (e.g., violence), autonomous driving (e.g., traffic accidents), etc. VAD is a challenging problem since anomalies are rare and longtailed in real life, leading to a lack of large-scale representative anomaly data.

⋆ This work was mostly done when Y. Yang was an intern at HRI-USA.

Fig. 1: Comparison of one-class VAD approaches. In this specific safety application example, only "walking" is normal. The test frame contains "skateboarding", so it is abnormal. (a) Traditional methods require full-shot training and only output anomaly scores, lacking reasoning. (b) Direct LLM use may not align with specific VAD needs. Here GPT-4V mistakenly treats "skateboarding" as normal. (c) Our AnomalyRuler has induction and deduction stages. It derives rules from few-shot normal reference frames to detect anomalies, correctly identifying "skateboarding" as an anomaly.

![Image](artifacts/image_000000_45eec9f6a4c0e414719a2dc30bd1db2c330ccb0ef9d2c84335de3b05040dde20.png)

Hence, the one-class VAD (a.k.a. unsupervised VAD) paradigm [16 , 36 , 43 , 45 , 53] is preferred, as it assumes that only the more accessible normal data are available for training. Most existing one-class VAD methods learn to model normal patterns via self-supervised pretext tasks, such as frame reconstruction [16 , 25 , 27 , 36 , 45 , 53 , 54] and frame order classification [14 , 43 , 49]. Despite good performance, these traditional methods can only output anomaly scores, providing little rationale behind their detection results (see Fig. 1a). This hinders them from earning public trust when deployed in real-world products.

We approach the VAD task with a reasoning framework toward a trustworthy system, which is less explored in the literature. An intuitive way is to incorporate the emergent Large Language Models (LLMs) [1 , 7 , 19 , 38 , 39 , 47], which have shown revolutionary capability in various reasoning tasks. Still, we find that their direct use falls short of performing VAD. Specifically, the implicit knowledge pre-trained in LLMs focuses on general context, meaning that it may not always align with specific real-world VAD applications. In other words, there is a mismatch between an LLM's understanding of anomalies and the anomaly definitions required for certain scenarios. For example, the GPT-4V [1] typically treats "skateboarding" as a normal activity, whereas certain safety applications need to define it as an anomaly, such as within a restricted campus (see Fig. 1b). However, injecting such specific knowledge by fine-tuning LLMs for each application is costly. This highlights the necessity for a flexible prompting approach that steers LLMs' reasoning strengths to different uses of VAD.

To arrive at such a solution, we revisit the fundamental process of the scientific method [4] emphasizing reasoning, which involves drawing conclusions in a rigorous manner [41]. Our motivation stems from two types of reasoning: inductive reasoning, which infers generic principles from given observations, and deductive reasoning, which derives conclusions based on given premises. In this

paper, we propose AnomalyRuler, a new VAD framework based on reasoning with LLMs. AnomalyRuler consists of an induction stage and a deduction stage as shown in Fig. 1c. In the induction stage, the LLM is fed with visual descriptions of few-shot normal samples as references to derive a set of rules for determining normality. Here we employ a Vision-Language Model (VLM) [23 , 51] to generate the description for each input video frame. Next, the LLM derives a set of rules for detecting anomalies by contrasting the rules for normality. The deduction, which is also an inference stage, follows the induced rules to identify anomalous frames in test video sequences. Additionally, in response to potential perception and reasoning errors by the VLM and LLM, we design strategies including rule aggregation via the randomized smoothing [10] for rule induction error mitigation, perception smoothing via the proposed Exponential Majority Smoothing for perception error reduction together with temporal consistency enhancement, and robust reasoning via a recheck mechanism for reliable reasoning output. These strategies are integrated into the AnomalyRuler pipeline to further enhance its detection robustness.

Apart from equipping VAD with reasoning ability, AnomalyRuler offers several advantages. First, AnomalyRuler is a novel few-normal-shot prompting approach that utilizes only a few normal samples from a training set as references to derive the rules for VAD. This avoids the need for expensive full-shot training or fine-tuning of the entire training set, as required by traditional one-class VAD methods. Importantly, it enables efficient adaption by redirecting LLM's implicit knowledge to different specific VAD applications through just a few normal reference samples. Second, AnomalyRuler shows strong domain adaptability across datasets, as the language provides consistent descriptions across different visual domains, e.g., "walking" over visual data variance. This allows the application of induced rules to datasets with similar scenarios but distinct visual appearances. Furthermore, AnomalyRuler is a generic framework that is complementary to VLM and LLM backbones. It accommodates both closed-source models such as the GPT family [1 , 39] and open-source alternatives such as Mistral [19]. To the best of our knowledge, the proposed AnomalyRuler is the first reasoning approach for the one-class VAD problem. Extensive experiments on four VAD datasets demonstrate AnomalyRuler's state-of-the-art performance, reasoning ability, and domain adaptability.

In summary, this paper has three main contributions. (1) We propose a novel rule-based reasoning framework for VAD with LLMs, namely AnomalyRuler. To the best of our knowledge, it is the first reasoning approach for one-class VAD. (2) The proposed AnomalyRuler is a novel few-normal-shot prompting approach that eliminates the need for expensive full-shot tuning and enables fast adaption to various VAD scenarios. (3) We propose rule aggregation, perception smoothing, and robust reasoning strategies for AnomalyRuler to enhance its robustness, leading to state-of-the-art detection performance, reasoning ability, and domain adaptability.

## 2 Related Work

Video Anomaly Detection. VAD is a challenging task since anomaly data are scarce and long-tailed. Therefore, researchers often focus on the one-class VAD (a.k.a. unsupervised VAD) paradigm [14 , 16 , 18 , 25 , 27 , 36 , 43 , 45 , 49 , 53 , 54], which uses only normal data during training. Most one-class methods learn to model normal patterns via self-supervised pretext tasks, based on the assumption that the model would obtain poor pretext task performance on anomaly data. Reconstruction-based methods [16 , 25 , 27 , 36 , 45 , 53 , 54] employ generative models such as auto-encoders and diffusion models to perform frame reconstruction or frame prediction as pretext tasks. Distance-based [14 , 43 , 49] methods use classifiers to perform pretext tasks such as frame order classification. These traditional methods can only output anomaly scores, providing little rationale behind their detection. Several recent studies explore utilizing VLMs or LLMs in anomaly detection. Elhafsi et al. [12] analyze semantic anomalies with an object detector [33] an LLM [7] in driving scenes. However, it relies on predefined concepts of normality and anomaly, which limits its adaption to different scenarios and cannot handle long-tailed undefined anomalies. Moreover, this method has not been evaluated on standard VAD benchmarks [2 , 22 , 24 , 28]. Cao et al. [8] explore the use of GPT-4V for anomaly detection, but their direct use may fall into the misalignment between GPT-4V's implicit knowledge and specific VAD needs, as discussed. Gu et al. [15] adopt a large VLM for anomaly detection, but it focuses on industrial images. Despite supporting dialogues, this method can only describe anomalies rather than explain the rationales behind its detection. Lv et al. [31] equip video-based LLMs in the VAD framework to provide detection explanations. It involves three-phase training to fine-tune the heavy video-based LLMs. Besides, it focuses on weakly-supervised VAD, a relaxed paradigm that requires training with anomaly data and labels. Different from these works, our AnomalyRuler provides rule-based reasoning via efficient few-normal-shot prompting and enables fast adaption to different VAD scenarios.

Large Language Models. LLMs [1 , 7 , 19 , 38 , 39 , 46 , 47] have achieved significant success in natural language processing and are recently being explored for computer vision problems. Recent advances, such as the GPT family [1 , 7 , 38 , 39], the LLaMA family [46 , 47], and Mistral [19], have shown remarkable capabilities in understanding and generating human language. On the other hand, large VLMs [1 , 21 , 23 , 34 , 44 , 51 , 57 , 58 , 60] have shown promise in bridging the vision and language domains. BLIP-2 [21] leverages Q-Former to integrate visual features into a language model. LLaVA [23] introduces a visual instruction tuning method for visual and language understanding. CogVLM [51] trains a visual expert module to improve large VLM's vision ability. Video-LLaMA [57] extends LLMs to understand video data. These models' parametric knowledge is trained for general purposes and thus may not apply to every VAD application. Recent studies explore prompting methods to exploit LLMs' reasoning ability. Chain-of-Thought (CoT) [11 , 52] guides LLMs to solve complex problems via multiple smaller and manageable intermediate steps. Least-to-Most (LtM) [20 , 59] decomposes a complex problem into multiple simpler sub-problems and solves them in sequence.

Fig. 2: The AnomalyRuler pipeline consists of two main stages: induction and deduction. The induction stage involves: i) visual perception transfers normal reference frames to text descriptions; ii) rule generation derives rules based on these descriptions to determine normality and anomaly; iii) rule aggregation employs a voting mechanism to mitigate errors in rules. The deduction stage involves: i) visual perception transfers continuous frames to descriptions; ii) perception smoothing adjusts these descriptions considering temporal consistency to ensure neighboring frames share similar characteristics; iii) robust reasoning rechecks the previous dummy answers and outputs reasoning.

![Image](artifacts/image_000001_5761d5255ed4dac996bd2f520f63f400b7ed0f7b3c0a8a734ea1d3576389083f.png)

Hypotheses-to-Theories (HtT) [61] learns a rule library for reasoning from labeled training data in a supervised manner. However, a reasoning approach for the VAD task in the one-class paradigm is not well-explored.

## 3 Induction

The induction stage aims to derive a set of rules from a few normal reference frames for performing VAD. The top part of Fig. 2 shows the three modules in the induction pipeline. The visual perception module utilizes a VLM which takes a few normal reference frames as inputs and outputs frame descriptions. The rule generation module uses an LLM to generate rules based on these descriptions. The rule aggregation module employs a voting mechanism to mitigate the errors from rule generation. In the following sections, we discuss each module and the strategies applied in detail.

## 3.1 Visual Perception

We design the visual perception module as the initial step in our pipeline. This module utilizes a VLM to convert video frames into text descriptions. We define Fn Fnormal = {fnormal 0 , . . . , fnormal n } as the few-normal-shot reference frames, with each frame fnormal i ∈ Fnormal randomly chosen from the training set. This module outputs the text description of each normal reference frame:

D normal = {VLM(fnormal i , pv) | fnormal i ∈ Fnormal}, with p v as the prompt "What are people doing? What are in the images other than people?". Instead of directly asking "What are in the image?", we design p v to separate humans and the environment with the following advantages. First, it enhances perception precision by directing the model's attention to specific aspects of the scene, ensuring that no details are overlooked. Second, it simplifies the following rule generation module by dividing the task into two subproblems [20], i.e., rules for human activities and rules for environmental objects. We denote this strategy as Human and Environment.

## 3.2 Rule Generation

With the text descriptions from normal reference frames Dnormal, we design a Rule Generation module that uses a frozen LLM to generate rules (denoted as R). In formal terms, R = {LLM(dnormal i , pg) | dnormal i ∈ Dnormal}, where p g is the prompt detailed in Appendix A.2. We craft p g with three strategies to guide the LLM in gradually deriving rules from the observed normal patterns:

Normal and Anomaly. The prompt p g guides the LLM to perform contrast , which first induces rules for normal based on D normal , which are assumed to be ground-truth normal. Then, it generates rules for anomalies by contrasting them with the rules for normal. For instance, if "walking" is a common pattern in D normal , it becomes a normal rule, and then "non-walking movement" will be included in the rules for anomaly. This strategy sets a clear boundary between normal and anomaly without access to anomaly frames.

Abstract and Concrete. The prompt p g helps the LLM to perform analogy , which starts from an abstract concept and then effectively generalizes to more concrete examples. Taking the same "walking" example, the definition of a normal rule is now expanded to "walking, whether alone or with others." Consequently, the anomaly rule evolves to include specific non-walking movements, i.e., "nonwalking movement, such as riding a bicycle, scooting, or skateboarding." This strategy clarifies the rules with detailed examples and enables the LLM to use analogy for reasoning without exhaustively covering every potential scenario.

Human and Environment. This strategy is inherited from the Visual Perception module. The prompt p g leads the LLM to pay attention separately to environmental elements (e.g., vehicles or scene factors) and human activities, separately. This enriches the rule set for VAD tasks, where anomalies often arise from interactions between humans and their environment.

These strategies align with the spirit of CoT [52] yet are further refined for the VAD task. The ablation study in Section 5.4 demonstrates their effectiveness.

## 3.3 Rule Aggregation

The rule aggregation module uses a LLM as an aggregator with a voting mechanism to combine n sets of rules (i.e., R) generated independently from n randomly chosen normal reference frames into one set of robust rules, Rrobust = LLM(R, p a ) .

This module aims to mitigate errors from previous stages, such as the visual perception module's potential misinterpretation of "walking" as "skateboarding", leading to incorrect rules. The aggregation process filters out uncommon elements by retaining rule elements consistently present across the n sets. The prompt p a for the LLM to achieve this is detailed in Appendix A.2. This strategy is based on the assumption of randomize smoothing [10], where errors may occur on a single input but are less likely to consistently occur across multiple randomly sampled inputs. Therefore, by aggregating these outputs, AnomalyRuler generates rules more resilient to individual errors. The hyperparameter n can be treated as the number of batches. For simplicity, previous discussions assume that each batch has only one frame, i.e., m = 1. Here we define m as the number of normal reference frames per batch, i.e., batch size. We show the effectiveness of the rule aggregation and provide an ablation on different n and m values in Section 5.4 .

## 4 Deduction

After the induction stage derives a set of robust rules, the deduction stage follows these rules to perform VAD. The bottom part of Fig. 2 illustrates the deduction stage, which aims to precisely perceive each frame of videos and then use the LLM to reason if they are normal or abnormal based on the rules. To achieve this goal, we design three modules. First, the visual perception module works similarly as described in the induction stage. However, instead of taking the few-normalshot reference frames, the deduction processes continuous frames from each test video and outputs a series of frame descriptions D = {d0, d1, . . . , dt}. Second, the perception smoothing module reduces errors with the proposed Exponential Majority Smoothing. This step alone can provide preliminary detection results, referred to as AnomalyRuler-base. Third, the robust reasoning module utilizes an LLM to recheck the preliminary detection results against the rules and perform reasoning. The perception smoothing and robust reasoning modules are elaborated in the following sections.

## 4.1 Perception Smoothing

As we discussed in Section 3.3, visual perception errors would happen in the induction stage, and this concern extends to the deduction stage as well. To address this challenge, we propose a novel mechanism named Exponential Majority Smoothing. This mechanism mitigates the errors by considering temporal consistency in videos, i.e., movements are continuous and should exhibit consistent patterns over time. We utilize the results of this smoothing to guide the correction of frame descriptions, enhancing AnomalyRuler's robustness to errors. There are four key steps:

Initial Anomaly Matching. For the continuous frame descriptions D = {d0, d1, . . . , dt}, AnomalyRuler first match anomaly keywords K found within the anomaly rules from the induction stage (see details in Appendix A.2), and assigns di with label yi where i ∈ [0, t], represents the predicted label. Formally,

we have yi = 1 if ∃k ∈ K ⊆ di, indicating an anomaly triggered by keywords such as ing-verb "riding" or "running". Otherwise, yi = 0 indicates the normal. We denote the initial matching predictions as Y = {y0, y1, . . . , yt} .

Exponential Majority Smoothing. We propose an approach that combines Exponential Moving Average (EMA) and Majority Vote. This approach is designed to enhance the continuity in human or object movements by adjusting the predictions to reflect the most common state within a specified window. The final smoothed predictions are denoted as Y ˆ = {yˆ ˆ 0, y ˆ 1, . . . , y ˆ t }, where each yˆ ˆ i is either 1 or 0. Formally, we have:

- Step I: EMA. For original prediction yt, the EMA value st is computed as st = P t i=0 (1−α) t − i P yi t i=0 (1−α) i . We denote α as the parameter that influences the weighting of data points in the EMA calculation.
- Step II: Majority Vote. The idea is to apply a majority vote to smooth the prediction within a window centered at each EMA value si with a padding size p. This means that for each si, we consider its neighboring EMA values within the window and determine the smoothed prediction yˆ ˆ i based on the majority of these values being above or below a threshold τ . We define this threshold as the mean of all EMA values: τ = 1 t P t i=1 si. Formally, the smoothed prediction y ˆ i is determined as:

<!-- formula-not-decoded -->

where 1(·) denotes the indicator function and the window size is adaptively defined as min(i + p, t) − max(1, i − p) + 1 ensuring that the window does not extend beyond the boundaries determined by the range from max(1, i − p) to min(i + p, t) .

Anomaly Score. Given that Y ˆ represents the initial detection results of AnomalyRuler, we can further assess these by calculating an anomaly score through a secondary EMA. Specifically, the anomaly scores, denoted as A = {a0, a1, . . . , at} , where a t is:

<!-- formula-not-decoded -->

We denote the above procedure AnomalyRuler-base as a baseline of our method, which provides a dummy answer, i.e., "Anomaly" if yˆ ˆ i = 1 otherwise "Normal", with an anomaly score that is comparable with the state-of-the-art VAD methods [3 , 25 , 35 , 43]. Subsequently, AnomalyRuler utilizes the dummy answer in the robust reasoning module for further analysis.

Description Modification. In this step, AnomalyRuler modifies the description D comparing Y and Y ˆ and outputs the modified D ˆ . If yi = 0 while yˆ ˆ i = 1 , indicating a false negative in the perception module, AnomalyRuler corrects di by adding "There is a person {k}.", where k ∈ K is the most frequent anomaly keyword within the window size w. Conversely, if yi = 1 while yˆ ˆ i = 0, indicating a false positive in the perception module, so AnomalyRuler modifies di by removing parts of the description that contain the anomaly keyword k .

## 4.2 Robust Reasoning

In the robust reasoning module, AnomalyRuler utilizes an LLM to achieve the reasoning task for VAD, with the robust rule Rrobust derived from the induction stage as the context. The LLM is fed with each frame's modified description ˆ d i with its dummy answer, i.e., either "Anomaly" or "Normal" generated from AnomalyRuler-base. We denote the output of robust reasoning as Y ∗ = {LLM( ˆ d i
, y ˆ i , Rrobust, p r ) | ˆ d i ∈ D, ˆ y ˆ i ∈ Y ˆ }. To ensure reliable results, the prompt p r , detailed in Appendix A.2, guides the LLM to recheck whether the dummy answer yˆ ˆ i matches the description ˆ di according to Rrobust. This validation step, instead of directly asking the LLM to analyze ˆ d i , improves decision-making by using the dummy answer as a hint. This approach helps AnomalyRuler reduce missed anomalies (false negatives) and ensures that its reasoning is more closely aligned with the rules. Additionally, to compare AnomalyRuler with the state-ofthe-art approaches based on thresholding anomaly scores, we apply Equation (2) with replacing yˆ ˆ i by y ∗ i ∈ Y ∗ to output anomaly scores.

## 5 Experiments

This section compares AnomalyRuler with LLM-based baselines and state-ofthe-art methods in terms of both detection and reasoning abilities. We also conduct an ablation study on each module within AnomalyRuler to evaluate their contributions. Examples of complete prompts, derived rules, and outputs are illustrated in Appendix A.2 .

## 5.1 Experimental Setup

Datasets. We evaluate our method on four VAD benchmark datasets. (1) UCSD Ped2 (Ped2) [22]: A single-scene dataset captured in pedestrian walkways with over 4,500 frames of videos, including anomalies such as skating and biking. (2) CUHK Avenue (Ave) [28]: A single-scene dataset captured in the CUHK campus avenue with over 30,000 frames of videos, including anomalies such as running and biking. (3) ShanghaiTech (ShT) [24]: A challenging dataset that contains 13 campus scenes with over 317,000 frames of videos, containing anomalies such as biking, fighting, and vehicles in pedestrian areas. (4) UBnormal (UB) [2]: An open-set virtual dataset generated by the Cinema4D software, which contains 29 scenes with over 236,000 frames of videos. For each dataset, we use the default training and test sets that adhere to the one-class setting. The normal reference frames used by AnomalyRuler are randomly sampled from the normal training set. The methods are evaluated on the entire test set if not otherwise specified. Evaluation Metrics. Following the common practice, we use the Area Under the receiver operating characteristic Curve (AUC) as the main detection performance metric. To compare with LLM-based methods that cannot output anomaly scores, we use the accuracy, precision, and recall metrics. Besides, we adopt the DoublyRight metric [32] to evaluate reasoning ability. All the metrics are calculated with frame-level ground truth labels.

Table 1: Detection performance with accuracy, precision, and recall (%) compared with different VAD with LLM methods on the ShT dataset.

| Method                           |   Accuracy  |   Precision  |   Recall |
|----------------------------------|-------------|--------------|----------|
| Ask LLM Directly                 |        52.1 |         97.1 |      6.2 |
| Ask LLM with Elhafsi et al. [12] |        58.4 |         97.9 |     15.2 |
| [ Ask Video-based LLM Directly  |        54.7 |         85.4 |      8.5 |
| AnomalyRuler                     |        81.8 |         90.2 |     64.3 |

Implementation Details. We implement our method, AnomalyRuler, using PyTorch [37]. If not otherwise specified, we employ CogVLM-17B [51] as the VLM for visual perception, GPT-4-1106-Preview [1] as the LLM for induction, and the open-source Mistral-7B-Instruct-v0.2 [19] as the LLM for deduction (i.e., inference) due to using GPTs on entire test sets is too costly. We discuss other VLMs/LLMs choices in Appendix A.4. The default hyperparameters of AnomalyRuler are set as follows: The number of batches in rule aggregation n = 10, the number of normal reference frames per batch m = 1, the padding size p = 5 in majority vote, and the weighting parameter α = 0 . 33 in EMA.

## 5.2 Comparison with LLM-based Baselines

Reasoning for one-class VAD using LLMs is not well-explored. To demonstrate AnomalyRuler's superiority over the direct LLM use, we build asking LLM/Videobased LLM directly as baselines and also adapt related works [8 , 12] to our target problem as baselines. At test time, let us denote test video frames as F = {f1, f2, . . . , ft}. We elaborate on our four baselines as follows. (1) Ask LLM Directly: {LLM(di, p) | di ∈ D}, where the LLM is Mistral-7B, D is F's frame descriptions generated by CogVLM, and p is "Is this frame description anomaly or normal?" (2) Ask LLM with Elhafsi et al. [12]: {LLM(di, p) | di ∈ D}, where the LLM is Mistral-7B, D is F's frame descriptions generated by CogVLM, and p is [12]'s prompts and predefined concepts of normality/anomaly. (3) Ask Video-based LLMs Directly: {Video-based LLM(ci, p) | ci ∈ C}, where p is "Is this clip anomaly or normal?" We use Video-LLaMA [57] as the Videobased LLM, which performs clip-wise inference. Each video clip ci consists of consecutive frames in F with the same label. (4) Ask GPT-4V with Cao et al. [8]: {GPT-4V(fi, p) | fi ∈ F}, where p is [8]'s prompts. As a large VLM, GPT-4V directly takes frames as inputs.

Detection Performance. Table 1 compares the accuracy, precision, and recall on the ShT dataset. Overall, AnomalyRuler achieves significant improvements with an average increase of 26.2% in accuracy and 54.3% in recall. Such improvements are attributed to the reasoning based on the rules generated in the induction stage. In contrast, the baselines tend to predict most samples as normal based on the implicit knowledge pre-trained in LLMs, resulting in very low recall and accuracy close to a random guess. Their relatively high precision is due to that they rarely predict anomalies, leading to fewer false positives.

Table 2: Reasoning performance with the Doubly-Right metric: {RR, RW, WR, WW} (%) on 100 (limited by GPT-4's query capacity) randomly selected frames from the ShT test set. We evaluate cases with visual perception errors (w. Perception Errors) and with manually corrected visual perception (w/o. Perception Errors).

| Method                             | w. Perception Errors    | w. Perception Errors    | w. Perception Errors    | w. Perception Errors    | . Perception Errors RW WR WW   | . Perception Errors RW WR WW   | . Perception Errors RW WR WW   | . Perception Errors RW WR WW   |
|------------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Method                             | RR                      | RW                      | WR                      | WW                      | RR                              | RW                              | WR                              | WW                              |
| Ask GPT-4 Directly                 | 57                      | 4                       | 15                      | 24                      | 73                              | 3                               | 0                               | 24                              |
| Ask GPT-4 with Elhafsi et al. [12] | 60                      | 3                       | 15                      | 22                      | 76                              | 2                               | 0                               | 22                              |
| Ask GPT-4V with Cao et al. [8]     | 74                      | 2                       | 7                       | 17                      | 81                              | 2                               | 0                               | 17                              |
| AnomalyRuler                       | 83                      | 1                       | 15                      | 1                       | 99                              | 0                               | 0                               | 1                               |

Reasoning Performance. The reasoning performance is evaluated using the Doubly-Right metric [32]: {RR, RW, WR, WW} (%), where RR denotes Right detection with Right reasoning, RW denotes Right detection with Wrong reasoning, WR denotes Right detection with Wrong reasoning, and WW denotes Wrong detection with Wrong reasoning. We desire a high accuracy of RR (the best is 100%) and low percentages of RW, WR and WW (the best is 0%). Since {RW, WR, WW} may be caused by visual perception errors rather than reasoning errors, we also consider the case with manually corrected visual perception to exclusively evaluate each method's reasoning ability, i.e., w. Perception Errors vs. w/o. Perception Errors in Table 2 .

Due to the lack of benchmarks for evaluating reasoning for VAD, we create a dataset consisting of 100 randomly selected frames from the ShT test set, with an equal split of 50 normal and 50 abnormal frames. For each frame, we offer four choices: one normal and three anomalies, where only one choice with the matched rules is labeled as RR, while the other choices correspond to RW, WR or WW. Details and examples of this dataset are illustrated in Appendix A.3 . Since the 100 randomly selected frames are not consecutive, here AnomalyRuler's perception smoothing is not used.

Table 2 shows the evaluation results. With perception errors, AnomalyRuler outperforms the baselines by 10% to 27% RR, and it achieves a very low WW of 1% compared to the 17% WW of the second best Ask GPT-4V with Cao et al. [8]. Without perception errors, AnomalyRuler's RR jumps to 99%. These results demonstrate AnomalyRuler's superiority over the GPT-4(V) baselines and its great ability to make correct detection along with correct reasoning.

## 5.3 Comparison with State-of-the-Art Methods

This section compares AnomalyRuler with 15 state-of-the-art one-class VAD methods across four datasets, evaluating their detection performance and domain adaptability. The performance values of these methods are sourced from their respective original papers.

Detection Performance. Table 3 shows the effectiveness of AnomalyRuler. There are three main observations. First, AnomalyRuler, even with its basic version AnomalyRuler-base, outperforms all the Image-Only competitors, which

Table 3: AUC (%) compared with different one-class VAD methods. "Image Only" methods only rely on image features. In contrast, others employ additional features such as bounding boxes from object detectors or 3D features from action recognition networks. "Training" indicates the methods that need a full-shot training process.

| Method            | Venue    | Image Only    | Training    | Ped2    | Ave    |   ShT  | UB   |
|-------------------|----------|---------------|-------------|---------|--------|--------|------|
| MNAD [36]         | CVPR-20  | ✓             | ✓           | 97.0    | 88.5   |   70.5 | -    |
| rGAN [29]         | ECCV-20  | ✓             | ✓           | 96.2    | 85.8   |   77.9 | -    |
| []  CDAE [9]     | ECCV-20  | ✓             | ✓           | 96.5    | 86.0   |   73.3 | -    |
| []  MPN [30]     | CVPR-21  | ✓             | ✓           | 96.9    | 89.5   |   73.8 | -    |
| []  NGOF [50]    | CVPR-21  | ✗             | ✓           | 94.2    | 88.4   |   75.3 | -    |
| [ HF2 [25]       | ICCV-21  | ✗             | ✓           | 99.2    | 91.1   |   76.2 | -    |
| []  BAF [14]     | TPAMI-21 | ✗             | ✓           | 98.7    | 92.3   |   82.7 | 59.3 |
| GCL [56]          | CVPR-22  | ✗             | ✓           | -       | -      |   79.6 | -    |
| S3R [53]          | ECCV-22  | ✗             | ✓           | -       | -      |   80.5 | -    |
| SSL [49]          | ECCV-22  | ✗             | ✓           | 99.0    | 92.2   |   84.3 | -    |
| zxVAD [3]         | WACV-23  | ✗             | ✓           | 96.9    | -      |   71.6 | -    |
| HSC [45]          | CVPR-23  | ✗             | ✓           | 98.1    | 93.7   |   83.4 | -    |
| FPDM [54]         | ICCV-23  | ✓             | ✓           | -       | 90.1   |   78.6 | 62.7 |
| SLM [43]          | ICCV-23  | ✓             | ✓           | 97.6    | 90.9   |   78.8 | -    |
| []  STG-NF [18]  | ICCV-23  | ✗             | ✓           | -       | -      |   85.9 | 71.8 |
| AnomalyRuler-base | -        | ✓             | ✗           | 96.5    | 82.2   |   84.6 | 69.8 |
| AnomalyRuler      | -        | ✓             | ✗           | 97.9    | 89.7   |   85.2 | 71.9 |

Table 4: AUC (%) compared with different cross-domain VAD methods. We follow the compared works to use ShT as the source domain dataset for other target datasets.

| Method            | Venue    | Image Only    | Training    |   Ped2  |   Ave  |   ShT1  | UB   |
|-------------------|----------|---------------|-------------|---------|--------|---------|------|
| rGAN [29]         | ECCV-20  | ✓             | ✓           |    81.9 |   71.4 |    77.9 | -    |
| [ MPN [30]       | CVPR-21  | ✓             | ✓           |    84.7 |   74.1 |    73.8 | -    |
| []  zxVAD [3]    | WACV-23  | ✗             | ✓           |    95.7 |   82.2 |    71.6 | -    |
| AnomalyRuler-base | -        | ✓             | ✗           |    97.4 |   81.6 |    83.5 | 65.4 |

1 AnomalyRuler employs UB as the source domain when ShT serves as the target domain. The competitors have no cross-domain evaluation on ShT, so we report their same-domain results.

also do not use any additional features (e.g., bounding boxes from object detectors or 3D features from action recognition networks), on the challenging ShT and UB datasets. This suggests that our rule-based reasoning benefits the challenging oneclass VAD task. Second, for Ped2 and Ave, AnomalyRuler performs on par with the Image-Only methods. This is achieved without any tuning, meaning that our few-normal-shot prompting approach is as effective as the costly full-shot training on these benchmarks. Third, AnomalyRuler outperforms AnomalyRuler-base, indicating that the robust reasoning module improves performance further.

Domain Adaptability. Domain adaptation considers the scenario that the source domain (i.e., training/induction) dataset differs from the target domain (i.e., testing/deduction) dataset [13 , 26 , 48]. We compare AnomalyRuler with three state-of-the-art VAD methods that claim their domain adaptation ability [3 , 29 , 30]. We follow the compared works to use ShT as the source domain dataset for other target datasets. As shown in Table 4, AnomalyRuler achieves the highest AUC on Ped2, ShT and UB, outperforming with an average of 9.88%. While AnomalyRuler trails zxVAD [3] by 0.6%, it is still higher than the others with an average of 8.85%. The results indicate that AnomalyRuler has better

domain adaptability across different datasets. This advantage is due to that the language provides consistent descriptions across different visual domains, which allows the application of induced rules to datasets with similar anomaly scenarios but distinct visual appearances. In contrast, traditional methods extract high-dimensional visual features that are sensitive to visual appearances, thereby struggling to transfer their knowledge across datasets.

## 5.4 Ablation Study

In this section, we look into how the proposed strategies affect AnomalyRuler. We investigate two aspects: rule quantity (i.e., the number of induced rules) and rule quality (i.e., their resulting performance). Regarding this, we evaluate variants of AnomalyRuler-base on the ShT dataset.

Ablation on Strategies. Table 5 shows the effects of removing individual strategies compared to using all strategies. In terms of rule quantity, removing Human and Environment or Normal and Anomaly significantly reduces rules by 47.6% and 82.4%, respectively. This reduction is due to not separating the rules for humans and the environment halves the number of rules. Moreover, without deriving anomaly rules from normal rules, we only have a limited set of normal rules. Removing Abstract and Concrete or Rule Aggregation slightly increases the number of rules, as the former merges rules within the same categories and the latter removes incorrect rules. Perception Smoothing does not affect rule quantity since it is used in the deduction stage. In terms of rule quality, removing Normal and Anomaly or Rule Aggregation has the most negative impact. The former happens because when only normal rules are present, the LLM overreacts to slightly different actions such as "walking with an umbrella" compared to the rule for "walking", leading to false positives. Furthermore, without rules for anomalies as a reference, the LLM easily misses anomalies. The latter is due to that perception errors in the induction stage would lead to incorrect rules for normal. Besides, removing other strategies also decreases AUC, underscoring their significance. In summary, the proposed strategies effectively improve AnomalyRuler's performance. There is no direct positive/negative correlation between rule quantity and quality, i.e., having too few rules leads to inadequate coverage of normality and anomaly concepts while having too many rules would cause redundancy and errors.

Ablation on Hyperparameters. Fig. 3 illustrates the effects of the hyperparameters in the rule aggregation and perception smoothing modules. For rule aggregation, we conduct cross-validation on the number of batches n = [1, 5, 10, 20] and the number of normal reference frames per batch m = [1, 2, 5, 10]. We observe that both the number of rules and AUC increase with the increases of n and m, but they start to fluctuate when n × m becomes large. For example, when n = 20, AUC drops from 85.9% to 72.2% as m increases because having too many reference frames (e.g., over 100) results in redundant information in a long context. For perception smoothing, we test the padding size in majority vote p = [1, 5, 10, 20] and the weighting parameter in EMA α = [0.09, 0.18, 0.33, 1]. We

Table 5: Ablation on strategies. We assess the effects of removing individual strategies in AnomalyRuler. We conduct the experiments five times with different randomly selected normal reference frames for induction and report their mean and standard deviation on the ShT dataset.

| Strategy                   | Stage       | # Rules A   | # Rules A   | # Rules A   | Accuracy Pr   | Accuracy Pr   | Accuracy Pr   | Recall    | Recall    | AUC   | AUC   |
|----------------------------|-------------|-------------|-------------|-------------|---------------|---------------|---------------|-----------|-----------|-------|-------|
| Strategy                   | Stage       | mean        | std         | mean        | std           | mean          | std           | mean      | std       | mean  | std   |
| w. All Below (default)     | Both        | 42.2        | 4.2         | 81.6        | 1.3           | 90.9          | 0.8           | 63.9      | 2.7       | 84.5  | 1.1   |
| w/o. Human and Environmen  | Both        | -20.1       | +1.         | -3.3        | +0.8          | -3.9          | +0.8          | -1.9      | +1.6      | -2.4  | +2.0  |
| w/o. Normal and Anomaly    | Induction - | -34.8       | 8 -1.3      | -20.5       | +4.3          | -41.2         | +7.0          | -14.4     | +11.6     | -18.8 | +1.2  |
| w/o. Abstract and Concrete | Induction   | +2.3        | 3 +2.       | -0.6        | -0.2          | -0.9          | -0.2          | -0.3      | -0.4      | -0.9  | +0.1  |
| w/o. Rule Aggregation      | Induction   | +8.5        | +6.1        | -9.6        | + 14.7        | +1.1          | +2.9          | -10.7     | +14.      | -15.8 | +0.8  |
| w/o. Perception Smoothing  | Deduction   | NA          | NA          | -1.7        | -0.9          | -1.9          | +0.1          | -3.8      | -0.3      | -3.3  | +0.8  |

Fig. 3: Ablation on hyperparameters of the (a) (b) rule aggregation and (c) perception smoothing modules on the ShT dataset.

![Image](artifacts/image_000002_2cc20761cf937dc10f006bc720788c05b8d39fb4d5fc75741595b708e4f603d8.png)

found p = 5 to be optimal for capturing the motion continuity in a video while avoiding the excessive noise that can occur with more neighborhoods. α adjusts the weight of the most recent frames compared to previous frames. A smaller α emphasizes previous frames, resulting in more smoothing but less responsiveness to recent changes. In general, increasing α from 0.09 to 0.33 improves AUC, suggesting that moderate EMA smoothing is beneficial.

## 6 Conclusion

In this paper, we propose AnomalyRuler, a novel rule-based reasoning framework for VAD with LLMs. With the induction and deduction stages, AnomalyRuler requires only few-normal-shot prompting without the need for expensive full-shot tuning, thereby fast steering LLMs' reasoning strengths to various specific VAD applications. To the best of our knowledge, AnomalyRuler is the first reasoning approach for one-class VAD. Extensive experiments demonstrate AnomalyRuler's state-of-the-art performance, reasoning ability, and domain adaptability. Limitations and potential negative social impact of this work are discussed in the Appendix A.1. In future research, we expect this work to advance broader oneclass problems and related tasks, such as industrial anomaly detection [6 , 55], open-set recognition [5 , 40], and out-of-distribution detection [17 , 42].

## Acknowledgments

This work was supported in part by National Science Foundation (NSF) under grants OAC-23-19742 and Johns Hopkins University Institute for Assured Autonomy (IAA) with grants 80052272 and 80052273. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of NSF or JHU-IAA.

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
2. Acsintoae, A., Florescu, A., Georgescu, M.I., Mare, T., Sumedrea, P., Ionescu, R.T., Khan, F.S., Shah, M.: Ubnormal: New benchmark for supervised open-set video anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022)
3. Aich, A., Peng, K.C., Roy-Chowdhury, A.K.: Cross-domain video anomaly detection without target domain adaptation. In: IEEE/CVF Winter Conference on Applications of Computer Vision (2023)
4. Bacon, F.: Novum organum (1620)
5. Bendale, A., Boult, T.E.: Towards open set deep networks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2016)
6. Bergmann, P., Fauser, M., Sattlegger, D., Steger, C.: Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019)
7. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: Conference on Neural Information Processing Systems (2020)
8. Cao, Y., Xu, X., Sun, C., Huang, X., Shen, W.: Towards generic anomaly detection and understanding: Large-scale visual-linguistic model (gpt-4v) takes the lead. arXiv preprint arXiv:2311.02782 (2023)
9. Chang, Y., Tu, Z., Xie, W., Yuan, J.: Clustering driven deep autoencoder for video anomaly detection. In: European Conference on Computer Vision (2020)
10. Cohen, J., Rosenfeld, E., Kolter, Z.: Certified adversarial robustness via randomized smoothing. In: International Conference on Machine Learning (2019)
11. Diao, S., Wang, P., Lin, Y., Zhang, T.: Active prompting with chain-of-thought for large language models. arXiv preprint arXiv:2302.12246 (2023)
12. Elhafsi, A., Sinha, R., Agia, C., Schmerling, E., Nesnas, I.A., Pavone, M.: Semantic anomaly detection with large language models. In: Autonomous Robots (2023)
13. Ganin, Y., Lempitsky, V.: Unsupervised domain adaptation by backpropagation. In: International Conference on Machine Learning (2015)
14. Georgescu, M.I., Ionescu, R.T., Khan, F.S., Popescu, M., Shah, M.: A backgroundagnostic framework with adversarial training for abnormal event detection in video. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2021)
15. Gu, Z., Zhu, B., Zhu, G., Chen, Y., Tang, M., Wang, J.: Anomalygpt: Detecting industrial anomalies using large vision-language models. In: AAAI Conference on Artificial Intelligence (2024)

16. Hasan, M., Choi, J., Neumann, J., Roy-Chowdhury, A.K., Davis, L.S.: Learning temporal regularity in video sequences. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2016)
17. Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-ofdistribution examples in neural networks. In: International Conference on Learning Representations (2017)
18. Hirschorn, O., Avidan, S.: Normalizing flows for human pose anomaly detection. In: IEEE/CVF International Conference on Computer Vision (2023)
19. Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., Casas, D.d.l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al.: Mistral 7b. arXiv preprint arXiv:2310.06825 (2023)
20. Lee, S., Kim, G.: Recursion of thought: A divide-and-conquer approach to multicontext reasoning with language models. In: Annual Meeting of the Association for Computational Linguistics (2023)
21. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International Conference on Machine Learning (2023)
22. Li, W., Mahadevan, V., Vasconcelos, N.: Anomaly detection and localization in crowded scenes. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2013)
23. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Conference on Neural Information Processing Systems (2023)
24. Liu, W., Luo, W., Lian, D., Gao, S.: Future frame prediction for anomaly detection– a new baseline. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018)
25. Liu, Z., Nie, Y., Long, C., Zhang, Q., Li, G.: A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction. In: IEEE/CVF International Conference on Computer Vision (2021)
26. Lo, S.Y., Oza, P., Chennupati, S., Galindo, A., Patel, V.M.: Spatio-temporal pixel-level contrastive learning-based source-free domain adaptation for video semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)
27. Lo, S.Y., Oza, P., Patel, V.M.: Adversarially robust one-class novelty detection. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)
28. Lu, C., Shi, J., Jia, J.: Abnormal event detection at 150 fps in matlab. In: IEEE/CVF International Conference on Computer Vision (2013)
29. Lu, Y., Yu, F., Reddy, M.K.K., Wang, Y.: Few-shot scene-adaptive anomaly detection. In: European Conference on Computer Vision (2020)
30. Lv, H., Chen, C., Cui, Z., Xu, C., Li, Y., Yang, J.: Learning normal dynamics in videos with meta prototype network. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021)
31. Lv, H., Sun, Q.: Video anomaly detection and explanation via large language models. arXiv preprint arXiv:2401.05702 (2024)
32. Mao, C., Teotia, R., Sundar, A., Menon, S., Yang, J., Wang, X., Vondrick, C.: Doubly right object recognition: A why prompt for visual rationales. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)
33. Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Dosovitskiy, A., Mahendran, A., Arnab, A., Dehghani, M., Shen, Z., et al.: Simple open-vocabulary object detection. In: European Conference on Computer Vision (2022)

34. Mittal, H., Agarwal, N., Lo, S.Y., Lee, K.: Can't make an omelette without breaking some eggs: Plausible action anticipation using large video-language models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)
35. Morais, R., Le, V., Tran, T., Saha, B., Mansour, M., Venkatesh, S.: Learning regularity in skeleton trajectories for anomaly detection in videos. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019)
36. Park, H., Noh, J., Ham, B.: Learning memory-guided normality for anomaly detection. In: IEEE/CVF Conference Computer Vision and Pattern Recognition (2020)
37. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. In: Conference on Neural Information Processing Systems (2019)
38. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al.: Improving language understanding by generative pre-training. OpenAI Blog (2018)
39. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI Blog (2019)
40. Safaei, B., Vibashan, V., de Melo, C.M., Hu, S., Patel, V.M.: Open-set automatic target recognition. In: IEEE International Conference on Acoustics, Speech and Signal Processing (2023)
41. Seel, N.M.: Encyclopedia of the sciences of learning (2011)
42. Sharifi, S., Entesari, T., Safaei, B., Patel, V.M., Fazlyab, M.: Gradient-regularized out-of-distribution detection. In: European Conference on Computer Vision (2024)
43. Shi, C., Sun, C., Wu, Y., Jia, Y.: Video anomaly detection via sequentially learning multiple pretext tasks. In: IEEE/CVF International Conference on Computer Vision (2023)
44. Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., Cai, D.: Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355 (2023)
45. Sun, S., Gong, X.: Hierarchical semantic contrast for scene-aware video anomaly detection. In: IEEE/CVF Computer Vision and Pattern Recognition Conference (2023)
46. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
47. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)
48. Tsai, Y.H., Hung, W.C., Schulter, S., Sohn, K., Yang, M.H., Chandraker, M.: Learning to adapt structured output space for semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018)
49. Wang, G., Wang, Y., Qin, J., Zhang, D., Bao, X., Huang, D.: Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles. In: European Conference on Computer Vision (2022)
50. Wang, H., Zhang, X., Yang, S., Zhang, W.: Video anomaly detection by the duality of normality-granted optical flow. arXiv preprint arXiv:2105.04302 (2021)
51. Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., et al.: Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079 (2023)
52. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. In: Conference on Neural Information Processing Systems (2022)

53. Wu, J.C., Hsieh, H.Y., Chen, D.J., Fuh, C.S., Liu, T.L.: Self-supervised sparse representation for video anomaly detection. In: European Conference on Computer Vision (2022)
54. Yan, C., Zhang, S., Liu, Y., Pang, G., Wang, W.: Feature prediction diffusion model for video anomaly detection. In: IEEE/CVF International Conference on Computer Vision (2023)
55. You, Z., Cui, L., Shen, Y., Yang, K., Lu, X., Zheng, Y., Le, X.: A unified model for multi-class anomaly detection. Conference on Neural Information Processing Systems (2022)
56. Zaheer, M.Z., Mahmood, A., Khan, M.H., Segu, M., Yu, F., Lee, S.I.: Generative cooperative learning for unsupervised video anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022)
57. Zhang, H., Li, X., Bing, L.: Video-llama: An instruction-tuned audio-visual language model for video understanding. In: Conference on Empirical Methods in Natural Language Processing (2023)
58. Zhang, Y., Huang, X., Ma, J., Li, Z., Luo, Z., Xie, Y., Qin, Y., Luo, T., Li, Y., Liu, S., et al.: Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514 (2023)
59. Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al.: Least-to-most prompting enables complex reasoning in large language models. In: International Conference on Learning Representations (2023)
60. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. In: International Conference on Learning Representations (2024)
61. Zhu, Z., Xue, Y., Chen, X., Zhou, D., Tang, J., Schuurmans, D., Dai, H.: Large language models can learn rules. arXiv preprint arXiv:2310.07064 (2023)

## A Appendix

## A.1 Limitations and Potential Negative Social Impact

Limitations. Similar to most existing LLM-based studies, AnomalyRuler assumes that the employed LLM backbones have decent capabilities. Sub-optimal LLMs may hinder the effectiveness of the methods. Exploring this limitation further could be an interesting future investigation.

Potential Negative Social Impact. The proposed method may enable malicious actors to more easily adapt VLMs/LLMs for illegal surveillance. To mitigate this risk, computer security mechanisms could be integrated.

## A.2 Examples of Input Prompts and Outputs Results

Induction. This stage starts from n randomly chosen normal reference frames Fn Fnormal = {fnormal 1 , . . . , fnormal n } and outputs a set of robust rules Rrobust. To simplify the illustration, we show one frame fnormal i ∈ Fnormal as an example in the visual perception and rule generation steps.

- Visual Perception
- – Input fnormal i and prompt p v :

![Image](artifacts/image_000003_c3bdd35de9fd44f622b08cdf6f3372e3f0a50dcef72d885f40e215cefd003ead.png)

pv = How many people are in the image and what is each of them doing ? What are in the image other than people ? Think step by step .

- – Outputs: Frame description dnormal i = V LM(fnormal i , pv)
- Rule Generation
- – Input prompt p g :

```
d normal i = There are four people in the image . Starting from the left , the first person is walking on the path . The second person is walking on the bridge . The third person is also walking on the bridge . The fourth person is also walking on the bridge . Other than people , there are trees , a railing , a path , and a bridge visible in the image .
```

```
pg = [ {' ' role ' ': '' system '', '' content ' ': As a surveillance monitor for urban safety using the ShanghaiTech dataset , my job is to derive rules for detecting abnormal human activities or environmental objects .} , {' ' role ' ': '' user ' ', '' content ' ': Based on the assumption that the given frame descriptions are normal , Please derive rules for normal , start from an abstract concept , and then generalize to concrete activities or objects .} , {' ' role ' ': '' assistant '', '' content ' ': ** Rules for Normal Human Activities : 1. ** Rules for Normal Environmental Objects : 1. }, {' ' role ' ': '' user ' ',
```

```
'' content ' ': Compared with the above rules for normal , can you provide potential rules for anomaly ? Please start from an abstract concept then generalize to concrete activities or objects , compared with normal ones .} , {' ' role ' ': '' assistant '', '' content ' ': ** Rules for Anomaly Human Activities : 1. ** Rules for Anomaly Environmental Objects : 1. }, {' ' role ' ': '' user ' ', '' content ' ': Now you are given frame description {dnormal i }. What are the Normal and Anomaly rules you have ? Think step by step . Reply following the above format , start from an abstract concept and then generalize to concrete activities or objects . List them using short terms , not an entire sentence .} , ]
```

- – Outputs: For each normal reference frame dnormal i , we will get one set of rules r i = LLM(dnormal i , pg). Since the structure of the rules is identical to the robust rules, we only present the robust rules in the following step as an illustration of our final induction output.
- Rule Aggregation
- – Input prompt p a :
- – Outputs: Robust rules Rrobust = LLM(R = {r1, . . . , r n }, p a )

```
pa = [ {' ' role ' ': '' system '', '' content ' ': As a surveillance monitor for urban safety using the ShanghaiTech dataset , my job is to organize rules for detecting abnormal activities and objects .} , {' ' role ' ': '' user ' ', '' content ' ': You are given {n} independent sets of rules for Normal and Anomaly . For the organized normal Rules , list the given normal rules with high - frequency elements For the organized anomaly Rules , list all the given anomaly rules }, {' ' role ' ': '' assistant '', '' content ' ': ** Rules for Anomaly Human Activities : 1. ** Rules for Anomaly Environmental Objects : 1. ** Rules for Normal Human Activities : 1. ** Rules for Normal Environmental Objects : 1. }, {' ' role ' ': '' user ' ', '' content ' ': Now you are given {n} independent sets of rules as the sublists of {R}. What rules for Anomaly and Normal do you get ? Think step by step , and reply following the above format .} , ]
```

```
** Rules for Anomaly Human Activities :** 1. Using any non - walking movement such as riding a bicycle or scooter , skateboarding 2. Running or jumping 3. Moving with the usual speed such as very fast or in a hurry 4. Person lying or bending down on the ground 5. Aggressive or unusual movements such as fighting or pushing 6. Loitering around the trash bin
```

```
7. Climbing on trees or into manhole / utility access points 8. Tampering with manhole covers 9. Engaging in physical altercation 10. Lingering without a clear purpose near a sensitive area 11. Falling unexpectedly ** Rules for Anomaly Environmental Objects :** 1. Missing , removed , or ajar manhole covers 2. Ground transportation such as vehicles , vans , bicycles . 3. Unattended bags or objects ( security threat ) 4. Unusual objects ( vehicles , large unattended luggage , etc .) 5. Unauthorized posters or signs 6. Damaged manhole cover or trees 7. Abandoned items in a public area 8. Yellow objects ( when not commonly seen ) 9. Graffiti on building windows or vandalism to a fence 10. Obstructions on pedestrian crossing 11. Fire hydrant spraying water without presence of emergency services 12. Unidentified objects near the tall structure 13. Smoke or fire coming from a utility access point 14. Objects thrown or falling from a height ** Rules for Normal Human Activities :** 1. Walking alone or with another person 2. Standing and using a mobile device 3. Walking on the sidewalk 4. Walking towards or away from a building 5. Entering a building 6. Standing near a trash bin 7. Waiting at traffic lights 8. Walking on pedestrian crossing 9. Sitting on bench or ground ** Rules for Normal Environmental Objects :** 1. Manhole covers on the ground or street 2. Trees along the street or Plants present 3. Trash bin in vicinity or placed beside the street 4. Posters on glass door 5. Static building with glass windows 6. Fence along the water 7. Pedestrian crossing markings visible 8. Standing fire hydrant 9. Static tall structure in the background 10. Utility access points on the ground
```

Deduction. This stage starts from a test video that contains t continuous frames F = {f1, . . . , ft} and outputs the reasoning results Y ˆ ∗ = {yˆ ˆ ∗ 1
, . . . , y ˆ ∗ t }. To simplify the illustration, we show two frames of this test video, fi, fj ∈ F as examples that represent one anomaly frame and one normal frame, respectively.

## · Visual Perception:

- – Input test frames fi , fj and prompt p v :
- – Outputs: Frame descriptions di = V LM(fi, p v ) , d j = V LM(fj , p v )

![Image](artifacts/image_000004_13cb8f4faab94474cee7f54d7163accd8c721903a0290036adaf2b3d0cbe760a.png)

## 22 Y. Yang et al.

- d i = There are four people in the image . One person is walking , another is also walking , the third person is riding a bicycle , and the fourth person is walking near the bicycle . Other than people , there are trees , a pathway , a trash bin , a bicycle , and two manhole covers visible in the image .

d j = There are two people in the image . One person appears to be walking , the other seems to be walking together . Other than people , there are two manhole covers on the ground , a trash bin , and some trees and plants .

## · Perception Smoothing:

- –
- Rrobust → K (generate anomaly keywords from anomaly rules, see Section 4).

![Image](artifacts/image_000005_6821bb3ec1dcd6591a66fe9ca69053f7a14ee6a3905240617030b50389d9963f.png)

## · Input prompt pk:

pk = You will be given a set of rules for detecting abnormal activities and objects ; please extract the anomaly keywords , activities using ''ing ' ' verbs , and anomaly objects using nouns , and provide a combined Python list with each represented by a single word . The output should be in the format : [" object1 " , ... , " activity1 ", " activity2 ", ...]. Now you are given {Rrobust} :

## · Output K:

```
anomaly_from_rule = [" trolley " , " cart " , " luggage ", " bicycle ", " skateboard ", " scooter ", " vehicles ", " vans ", " accident " , " running ", " jumping ", " riding " , " skateboarding ", " scooting ", " lying ", " falling ", " bending ", " fighting ", " pushing ", " loitering ", " climbing ", " tampering " , " lingering "]
```

- – d i → ˆ d i &amp; yˆ ˆ i
, d j → ˆ d j &amp; yˆ ˆ j (generate modified descriptions and smoothed predictions, see Section 4).

## · Robust Reasoning:

- – Input prompt p r :

```
pr = You will be given an description of frame , your task is to double check my initial anomaly detection result based on the rules . The rules are : {R robust } My initial result is {yˆ ˆ i } First , if human activity is present , which rule is matching ? List the rule category , e.g. , normal or anomaly , with the rule number . Second , if an environmental object is present , which rule is matching ? List the rule category , e.g ., normal or anomaly , with the rule number . Third , are the human activities or environmental objects anomaly ? Answer : anomaly , if ANY anomaly rule ( even if only one , no matter human activities or environmental objects ) matches , otherwise answer : normal . Now you are given the frame {d ˆ i }, think step by step .
```

## – Final reasoning output: y ∗ i = LLM( ˆ d i
, y ˆ i , Rrobust, p r )

```
y ˆ ∗ i = The third person is riding a bicycle , which matches rule number 1 under Anomaly Human Activities . The bicycle matches rule number 2 under Anomaly Environmental Objects . Answer : anomaly .
```

```
y ∗ j = LLM( ˆ d j , y ˆ j , Rrobust, p r ) y ˆ ∗ j = The human activities present are two people walking . The rules for normal human activities include walking alone or with another person . So , the rule matching is : 1. Walking alone or with another person ( normal ) The environmental objects present are two manhole covers on the ground , a trash bin , and some trees and plants . So , the rules matching are : 1. Manhole covers on the ground or street ( normal ) 2. Trees along the street or Plants present ( normal ) 3. Trash bin in vicinity or placed beside the street ( normal ) Answer : normal .
```

## A.3 Examples of Doubly-Right Reasoning Evaluation

In Section 5.2, we evaluate reasoning ability using the Doubly-Right metric [32]. We create a benchmark dataset with multiple choices for Doubly-Right reasoning evaluation. The evaluation is conducted in the deduction stage, where we input the visual perception description and the induced rules to an LLM. The goal is to demonstrate that the induced rules enable LLMs to perform correct reasoning.

We list the prompt for reasoning evaluation below and one example of the description and its four choices as Table 6. The content in normal choice is fixed, while the anomaly choices include one correct reasoning with a matched rule and two randomly chosen non-matched rules from our generated anomaly rules. In this example, Choices A, B, C and D correspond to RW, WW, RR and RW, respectively.

```
[ {' ' role ' ': '' system '', '' content ' ': You will be given a description of the frame and four choices . Your task is to make the correct choice based on the rules . The rules are : {R robust }} , {' ' role ' ': '' user ' ', '' content ' ': Description : {d ˆ i } Choices : { Four Choices } Choose just one correct answer from the options (A , B , C , or D) and output without any explanation . Please Answer :} , ]
```

## A.4 Different VLMs/LLMs as Backbones

Table 7 shows the results of using various VLMs/LLMs as backbones in the deduction stage, compared to the default setting (the first row). All the results are based on the same rules derived in the induction stage with the default setting.

We categorize the comparisons into three types: (1) VLMs only: AnomalyRuler uses the same VLMs as an end-to-end solution, combining visual perception and robust reasoning. It inputs the test frame and outputs the reasoning result. This category includes GPT-4V [1], LLaVA [23], and PandaGPT [44]. (2) VLMs + Mistrial [19]: We keep Mistrial as the default LLM for robust reasoning and test different VLMs (e.g., OWLViT [33], LLaVA, BLIP-2 [21], RAM [58]) for visual

Table 6: An example of reasoning performance evaluation with multiple reasoning choices. In this example, Choices A, B, C and D correspond to RW, WW, RR and RW of the Doubly-Right metric, respectively. The RR choice is highlighted in yellow .

| Frame Description                                    | Multiple Choices for Reasoning Evaluation                                                           |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| There are four people in the imA.                   | he imA. Anomaly, since “climbing on a tree” matches anomaly human                                  |
| age. One pe                                          | g withactivities “Climbing on trees or into manhole/utility access points”.                         |
| a backpack, an                                       | her person isB. Normal, since no rules for anomaly human activities or non                         |
| riding a bicycle, a th                               | another person is le, a third person B. Normal, since no rules for anomaly  human objects match. |
| is standing and looking at                           | and looking at the d hfh  C. Anomaly, since “riding a bicycle” matches anomaly human ac “lkh dbl |
| bicyclist, and the four                              | nd the fourth persontivities “Using any non-walking movement such as riding a bicycle               |
| is sitting on a bench. Other thanor scooter, skateb  | her thanor scooter, skateboarding”.                                                                 |
| people, there are trees                              | re trees, a trashD. Anomaly, since “a vehicle parked blocking a pedestrian crossing”                |
| bin, and two manh                                    | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| is sitting on a bench. Other thanor scooter, skatebo | ther thanor scooter, skateboarding”.                                                                |
| is sitting on a bench. Other thanor scooter, skatebo | Other thanor scooter, skateboarding”.                                                               |
| people, there are trees, a trashD                    | trees, a trashD. Anomaly, since “a vehicle parked blocking a pedestrian crossing”                   |
| people, there are trees, a trashD. Ano               | rees, a trashD. Anomaly, since “a vehicle parked blocking a pedestrian crossing”                    |
| bin, and two manhole covers                          | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole covers vis                     | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| ,  ible in the image.                               | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image                                    | crossing”.                                                                                          |
| bicyclist, and the fourth persontiviti               | the fourth persontivities “Using any non-walking movement such as riding a bicycle                  |
| ible in the image.                                   | p h. Other than g y g g y or scooter, skateboarding”.                                            |
| bin, and two manhol ible in the image               | satces aoay ouaobjects Obstuctos opedesta crossing”.                                               |
| bin, and two manh ible in the image                 | matches anomaly nonhuman objects Obstructions on pedest crossing”                                  |
| ,  ible in the image.                               | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| pp,  bin, and two manhole                           | crossing”.                                                                                          |
| bin, and two manhole covers vismat                  | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole covers vismatc                 | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole covers vismat                  | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole covers vis                      | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole covers vismatches              | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole covers vis                      | covers vismatches anomaly non-human objects “Obstructions on pedestrian                            |
| bin, and two manhole                                 | y j rossing”.                                                                                      |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | c                                                                                                   |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | ble in the image.                                                                                   |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”.                                                                                          |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |
| ible in the image.                                   | crossing”                                                                                           |

Table 7: Detection performance with accuracy, precision, and recall (%) using different VLMs/LLMs as backbones in the deduction stage on 100 (limited by GPT-4's query capacity) randomly selected frames from the ShT test set.

| Visual Perception     | Robust Rea           |   Accuracy  |   Precision  |   Recall  | Open Sourc   |
|-----------------------|----------------------|-------------|--------------|-----------|--------------|
| CogVLM [51] (default) | Mistral [19] (defaul |          82 |         88.1 |        74 | ✓            |
| GPT-4V [1]            | GPT-4V               |          83 |         88.4 |        76 | ✗            |
| LLaVA [23]            | LLaVA                |          40 |         40.4 |        42 | ✓            |
| PandaGPT [44]         | PandaGPT             |          37 |         31.4 |        22 | ✓            |
| OWLViT [33]           | Mistral              |          71 |         82   |        54 | ✓            |
| LLaVA                 | Mistral              |          76 |         79.5 |        70 | ✓            |
| BLIP-2 [21]           | Mistral              |          50 |         50   |        94 | ✓            |
| RAM [58]              | Mistral              |          45 |         47.2 |        84 | ✓            |
| CogVLM                | GPT-3.5 [7]          |          81 |         86   |        74 | ✗            |
| CogVLM                | LLaMA-2 [47]         |          60 |         70.8 |        34 | ✓            |

perception. (3) CogVLM [51] + LLMs: We use CogVLM as the fixed VLM for visual perception and test different LLMs for robust reasoning (e.g., GPT-3.5 [7], LLaMA-2 [47]). We have the following observations.

For the VLMs-only category, GPT-4V performs well, but it has limitations on the number of queries and a high cost per query, making it expensive for largescale testing. LLaVA and PandaGPT, on the other hand, show poor reasoning ability. They cannot follow the provided robust rules, and generate irrelevant content or hallucinations. An example frame with their outputs is shown below:

![Image](artifacts/image_000006_00c232cb34c2b736d7667cc61b1dbb6d8e5754f83aa4c16019986876effec819.png)

For the VLMs + Mistrial category, using OWLViT and LLaVA as visual perception modules yields usable results, though they are still 6 to 10% lower than using CogVLM. However, the results with BLIP-2 and RAM are not usable due to serious hallucinations. For example, in a normal frame featuring only

people walking, BLIP-2 outputs "A sidewalk with trees, two people are walking down a sidewalk, a man is riding a skateboard on a sidewalk, a woman walking down a sidewalk in a park.", while RAM (recognize anything) outputs "Image Tags: path | person | skate | park | pavement | plaza | skateboarder | walk".

For the CogVLM + LLMs category, GPT-3.5 performs well but is expensive for large-scale testing. LLaMA-2, on the other hand, struggles with reasoning and fails to follow the given rules as context effectively.

In summary, the propose AnomalyRuler is a generic plug-and-play framework that can improve VAD performance upon both the closed-source GPTs and the open-source VLMs/LLMs such as CogVLM and Mistral. AnomalyRuler applies to various VLMs/LLMs backbones as long as they have decent visual perception and rule-following capabilities.

## A.5 Further Discussions on Perception Smoothing and Robust Reasoning

Sections 5.3 and 5.4 demonstrate the effectiveness of the proposed perception smoothing and robust reasoning strategies. In this section, we provide a deeper investigation into them. Specifically, we aim to examine the extent to which the smoothing step may incorrectly smooth out anomalies from a sequence of video frames, and the extent to which the robust reasoning step can rectify these errors.

Table 8 shows that less than 0.7% of anomalies are incorrectly smoothed out by the perception smoothing step (before the robust reasoning step), indicating very low false negative rates. The subsequent robust reasoning step successfully rechecks and corrects inaccuracies in the smoothed results, further reducing the false negative rates to below 0.15%.

Table 8: The percentage (%) of incorrectly smoothed-out anomalies by the perception smoothing strategy on each dataset.

| Dataset                 | ShT    | Ave    | Ped2    | UB    |
|-------------------------|--------|--------|---------|-------|
| Before Robust Reasoning | 0.7%   | 0.4%   | 0.6%    | 0.3%  |
| After Robust Reasoning  | 0.08%  | 0.15%  | 0.08%   | 0.01% |

The low false negative rates are due to that the smoothing step only smooths out the brief, isolated frames within a sequence of continuous frames. Table 9 shows that brief anomalies are rare in VAD datasets, as they typically persist for 97.9 to 441.3 continuous frames due to the time required for an anomaly to enter and exit the camera's view. We also calculated the percentage of brief frames, i.e., ≤ 10 frames, among all continuous anomaly frames. The ShT dataset has the highest percentage at 17.5% and an average length of 5.5 frames. In Section 5.4 , we find that a padding size p = 5 in our majority vote step is the optimal window size for ShT for capturing the predominant motion continuity in a video. This aligns with the average length of brief continuous anomalies (5.5 frames) and may explain the reason behind this optimal value.

Table 9: Statistics for the number of continuous anomaly frames per video clip of each dataset.

| Dataset                                    | ShT    | Ave    | Ped2    | UB    |
|--------------------------------------------|--------|--------|---------|-------|
| # Average continuous anomaly frames        | 111.3  | 97.9   | 137.3   | 441.3 |
| % Brief continuous anomalies (≤ 10 frames) | 17.5%  | 2.1%   | 0.0%    | 0.0%  |
| # Average brief continuous anomaly frames  | 5.5    | 10.0   | 0.0     | 0.0   |

## A.6 Normal Reference Frame Sampling

The proposed few-normal-shot prompting method is particularly beneficial when only a few normal data points are available in real-world scenarios. In our experiments, we simulate this scenario by randomly sampling normal reference frames from a training set, assuming only the randomly sampled frames are available.

However, even when a set of normal data (e.g., a training set) has already been collected, our few-normal-shot prompting method is still useful for fast adaptation. In this scenario, different normal reference frame sampling strategies beyond random sampling can be considered, such as sampling by GPT-4V [1]. Table 10 compares the random sampling and GPT-4V sampling (sampling ten frames) on the ShT dataset. The results of five trials show similar performance. The reason is that normal patterns in existing VAD datasets are not very diverse. Hence, randomly sampled normal frames are efficient as references for rule induction. Requiring only a few randomly sampled reference frames is one of our contributions, but GPT-4V sampling could be a promising extension for more complicated VAD scenarios.

Table 10: Random sampling vs. GPT-4V sampling on the ShT dataset. Results of five trials are reported.

| Method                       | # Rules    | AUC (%)    |
|------------------------------|------------|------------|
| Random sampling (ten frames) | 42.2 ± 4.2 | 84.5 ± 1.1 |
| GPT-4V sampling (ten frames) | 39.9 ± 6.9 | 84.8 ± 1.6 |

## A.7 Unified Anomaly Detection

Unified anomaly detection [55] considers image anomaly detection that trains a single model across different object classes. We extend this setting to VAD by considering a single model across different datasets. Specifically, the proposed AnomalyRuler can perform as a unified anomaly detection approach by using normal reference frames randomly sampled from various datasets and deriving a set of unified rules for all datasets. Table 11 shows the results, which are on par with the main evaluation in Table 3. This demonstrates that AnomalyRuler performs well under the unified anomaly detection setting by inducing effective unified rules across datasets with similar anomaly scenarios but distinct visual appearances.

Table 11: AUC (%) of AnomalyRuler under the unified anomaly detection setting. AnomalyRuler induces unified rules from a few normal reference frames across all four datasets and is evaluated on these datasets.

|   Ped2  |   Ave  |   ShT  |   UB |
|---------|--------|--------|------|
|    97.6 |   85.6 |   84.7 | 68.8 |