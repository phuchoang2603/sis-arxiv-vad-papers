---
title: 'A Survey on Video Anomaly Detection via Deep Learning: Human, Vehicle, and
  Environment'
type: survey
categories:
- Semi Supervised
- Unsupervised
- Instruction Tuning
- Hybrid
github_link:
description: This survey provides a comprehensive overview of deep 
  learning-based Video Anomaly Detection (VAD), covering challenges, 
  methodologies, domain-specific applications, and future research directions 
  across human-centric, vehicle-centric, and environment-centric contexts. It 
  introduces a taxonomy of supervision levels, adaptive learning strategies, and
  explores diverse application areas including healthcare, public safety, road 
  surveillance, and disaster detection, emphasizing the latest advancements and 
  open challenges.
benchmarks:
- cuhk-avenue
- shanghaitech
- xd-violence
- ucf-crime
- ucsd-ped
- other
authors:
- Ghazal Alinezhad Noghre
- Armin Danesh Pazho
- Hamed Tabkhi
date: '2023-10-01'
---

## A Survey on Video Anomaly Detection via Deep Learning: Human, Vehicle, and Environment

Ghazal Alinezhad Noghre 1 , Armin Danesh Pazho 1 , Hamed Tabkhi 1

Abstract—Video Anomaly Detection (VAD) has emerged as a pivotal task in computer vision, with broad relevance across multiple fields. Recent advances in deep learning have driven significant progress in this area, yet the field remains fragmented across domains and learning paradigms. This survey offers a comprehensive perspective on VAD, systematically organizing the literature across various supervision levels, as well as adaptive learning methods such as online, active, and continual learning. We examine the state of VAD across three major application categories: human-centric, vehicle-centric, and environment-centric scenarios, each with distinct challenges and design considerations. In doing so, we identify fundamental contributions and limitations of current methodologies. By consolidating insights from subfields, we aim to provide the community with a structured foundation for advancing both theoretical understanding and real-world applicability of VAD systems. This survey aims to support researchers by providing a useful reference, while also drawing attention to the broader set of open challenges in anomaly detection, including both fundamental research questions and practical obstacles to real-world deployment.

Index Terms—video anomaly detection, deep learning, computer vision

## I. INTRODUCTION

V IDEO Anomaly Detection (VAD), also known as outlier detection, abnormal event detection, and abnormal activity detection, has emerged as a crucial technology across a range of applications [1]–[5], from public safety [6]–[10] to healthcare monitoring [11]–[14], autonomous driving [15]– [18], road surveillance [19]–[21], and environmental disaster detection and response [22]–[27]. In this age, where thousands of cameras continuously capture data, automated systems for detecting unusual events offer transformative potential [28]– [31]. For example, surveillance VAD (see Supplementary Materials for list of abbreviations) can automatically flag crimes or accidents, relieving human operators of the impossible task of watching hours of mostly uneventful footage. Another example is in healthcare, where VAD can monitor patients or older adults for sudden falls or distress. The growing importance of VAD in such domains stems from its ability to consistently watch for anomalies that could signify security threats, medical emergencies, or catastrophic events.

VAD confronts unique challenges inherent to video data. A video is a high-dimensional spatiotemporal signal: each anomaly may involve not just an unusual appearance in a single frame, but an irregular motion pattern unfolding over time [32]–[34]. An anomaly in video can be formally defined as "The manifestation of atypical visual or motion characteristics,

1 Electrical and Computer Engineering Department, UNC Charlotte (galinezh, adaneshp, htabkhiv@charlotte.edu)

or the presence of typical visual or motion patterns occurring in a spatiotemporal contexts that deviate from established norms". An example of an abnormal pattern can be a car accident, which represents a deviation from expected vehicular operation. On the other hand, a normal pattern occurring in an inappropriate context is exemplified by riding a bicycle on a pedestrian-only sidewalk. Moreover, regardless of the specific domain or application area, anomalous events are inherently rare, often occurring with low frequency and unpredictability [8], [35]–[39]. VAD may encounter novel, unforeseen abnormal events that were never observed. Even new patterns of normal activity may continually emerge, especially in openworld environments [40]–[43].

Traditionally, VAD relied on statistical models and handcrafted features to identify unusual patterns [44], [45]. These methods often struggled with the complexity and variability inherent in video data and previously mentioned general challenges, leading to limited accuracy and adaptability. Deep learning has been a driving force behind recent progress in VAD, enabling models to automatically learn rich representations of normal and abnormal patterns. A wide spectrum of learning paradigms, from fully supervised [21], [46]–[51] to unsupervised [32], [52]–[55], has been explored in the literature. Beyond the training data regime, researchers have also looked at adaptive learning paradigms for VAD [43], [56], [57]. The abundance of these paradigms reflects the community's efforts to tackle VAD's challenges from different angles. Each paradigm comes with its own assumptions, strengths, and failure modes, and part of the goal of our survey is to clarify how these pieces fit into the larger picture.

Considering the breadth of applications and methods, there is a clear need for a unifying, structured perspective on VAD. Past surveys have typically focused on a subset of this space [2], [4], [6]–[9], [11]–[20], [23], [58]–[62]. For instance, on algorithms for a single domain (e.g. autonomous vehicles) or on specific training paradigms (e.g., unsupervised anomaly detection). However, VAD research has now grown to encompass diverse domains and a wide array of deep learning techniques. Researchers in one domain may not be fully aware of relevant techniques developed in another domain, even though the underlying problems share similarities. We aim to bridge this gap by providing a comprehensive survey that treats VAD holistically. In particular, we bring together humancentric, vehicle-centric, and environment-centric VAD under one umbrella (see Figure 1). By comparing and contrasting the problem formulations, data characteristics, and successful techniques across these domains, our survey highlights common principles as well as domain-specific nuances. This

1

Fig. 1. Overview of the paper structure. The advancements in vehicle, human, and environmental VAD are explored.

![Image](artifacts/image_000000_14ccb66c43f2f04350a14673972df056e903afa856bc270fbdbb7220d5782992.png)

unified viewpoint is intended to help transfer knowledge across application domains. Moreover, we organize the growing literature on deep learning for VAD into a coherent taxonomy, which makes it easier to understand how different approaches relate to each other. Rather than seeing the field as a collection of disjoint research focuses, readers will gain a structured map of VAD research: the key problem settings, the algorithmic families, and the connections between them. We aim to help the research community identify open problems and practical barriers that must be addressed to advance VAD towards widespread deployment. This survey aims to bring clarity to what has been accomplished and what remains to be done. We summarize not only the State-of-The-Art (SOTA) techniques but also their limitations, and we pinpoint areas ready for new exploration. To this end, the main contributions of this paper are as follow:

- We identify and critically analyze key challenges and open problems in VAD. By highlighting these gaps, the survey outlines practical considerations necessary for building reliable, adaptive, and deployable VAD systems.
- We present a structured taxonomy of VAD approaches categorized by supervision levels and learning paradigms, including supervised, unsupervised, weakly supervised, self-supervised, and adaptive learning. This taxonomy clarifies the underlying assumptions, strengths, and limitations of each paradigm and guides readers in selecting appropriate methods for different problem settings.
- We provide a comprehensive and unified survey of VAD via deep learning, encompassing human-centric, vehiclecentric, and environment-centric domains. This work bridges the gap between fragmented subfields by systematically comparing problem formulations, data characteristics, and methods across application areas, enabling knowledge transfer and cross-domain insights.

## II. VAD CHALLENGES

VAD presents unique challenges, summarized in Table I, which are explained in detail in this section.

## A. Data Scarcity and Annotation Challenges

- 1) C1: Rarity of Anomalies and Class Imbalance: By definition, anomalies are rare events compared to normal. For instance, traffic accidents in autonomous driving are infrequent compared to normal driving scenarios. Deep learning models

typically thrive on abundant data, but the scarcity of anomalous examples means they struggle to learn generalizable patterns.

- 2) C2: Limited and Difficult Labeling: Not only are anomalies rare, but they are also inherently difficult to label. Annotating frame-by-frame or pixel-level ground truth is labor-intensive. In many instances, expert knowledge is also essential to perform correct labeling. For example, in healthcare applications like monitoring Parkinson's disease, identifying the exact onset and offset of anomalous behavior necessitates the involvement of domain specialists.
- 3) C3: Ambiguity in Defining Anomalies and Context Specificity: Unlike standard vision tasks, anomaly detection is highly context-dependent. The same behavior may be normal in one setting but anomalous in another. For instance, in public safety, punching signals violence unless occurring in a boxing gym. Such contextual ambiguity complicates defining anomalies. Modeling such context is difficult and requires auxiliary inputs or learning multiple modes of normality. While deep models must be robust to these ambiguities, current methods often struggle with subtle or context-sensitive anomalies.

## B. Spatiotemporal Modeling Challenges

- 1) C4: Complex Temporal Patterns and Long-Term Dependencies: Anomalies unfold as irregular motion patterns or unusual events over time. Capturing temporal dynamics is a core difficulty. For instance, in Autonomous Driving, an accident might be inferred from a vehicle's erratic trajectory over several seconds. Some anomalies have a slow temporal build-up (e.g., a person slowly loitering in a restricted area). Detecting these requires integrating information over long durations. On the other hand, anomalies can be instantaneous (a sudden explosion). Balancing responsiveness to quick events with the ability to analyze extended sequences is non-trivial.
- 2) C5: Multi-Agent Interactions and Crowded Scenes: Many anomalies involve multiple entities interacting with each other. Detecting these anomalies requires modeling collective behavior patterns. However, modeling them is difficult due to occlusions and complex dynamics. In some events the anomaly is evident in the group's joint configuration (e.g. a group of people suddenly running away) even if each individual's motion by itself might appear normal.
- 3) C6: Feature Abstraction Level: Deep video anomaly detectors traditionally operate on raw pixel data, but this raises feature redundancy issues. Raw pixel-based models

TABLE I COMPREHENSIVE SUMMARY OF KEY CHALLENGES IN VIDEO ANOMALY DETECTION. THE TABLE CATEGORIZES THE CHALLENGES INTO SIX BROAD THEMES. THIS CATEGORIZATION AIMS TO GUIDE FUTURE RESEARCH AND DEVELOPMENT DIRECTIONS IN VIDEO ANOMALY DETECTION SCENARIOS .

| Category                                 | Challeng                                                                                                           | Short Description                                                                                                                      | References                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| 1: Rarity of Anomalies 
 Class Imbalance | C1: Rarity of Anomalies and
 Class Imbalance                                                                       | [8], [35], [36]                                                                                                                        | Data Scarcity and
 Annotation
 Chll |
| 1: Rarity of Anomalies 
 Class Imbalance | C2: Limited and Difficult Labeling                                                                                 | e annotation is labor-intensive and often require
 domain experts                                                                      | [40], [63], [64]                    |
| 1: Rarity of Anomalies 
 Class Imbalance | C3: Ambiguity in Defining
 Anomalies and Context Specificity                                                       | omalies are context-dependent; similar actions ca
 be normal or abnormal depending on scenario. 
 liifi                                | [64]–[66]                           |
| Spatiotemporal
 Modeling
 Challenges     | C4: Complex Temporal Patterns
 and Long-Term Dependencies                                                          | Many anomalies manifest over time. Instantaneous and
 rolonged anomalies must be detected with appropriate
 temporal context.
 ilfilid | [32]–[34                            |
| Spatiotemporal
 Modeling
 Challenges     | C5: Multi-Agent Interactions and
 Crowded Scenes                                                                   | eractions are complex, often involving crowd
 behavior or occlusion.                                                                   | [6], [67], [68                      |
| Spatiotemporal
 Modeling
 Challenges     | C6: Feature Abstraction Leve                                                                                       | -based models are affected by visual noise. Higher
 abstraction may lose contextual cues.                                              | [9], [41], [43], [69], [70          |
| Robustness and
 Generalization           | C7: Environmental Variations and
 Noise
 C8DiShifd                                                                 | orld conditions (e.g., weather, lighting) degrade
 model performance. 
 dlffil hdld iil                                                | [65], [71], [72]                    |
| Robustness and
 Generalization           | C8: Domain Shift and
 Cross-Scene Generalization                                                                   | ten fail when deployed in new v
 environments. 
 f l/lb                                                                                | [73]–[75]                           |
| Robustness and
 Generalization           | C9: Open-Set Nature of
 Anomalies and Novelty                                                                      | ot all types of normal/anomaly can be seen during
 training or validation.                                                             | [40]–[42                            |
| Robustness and
 Generalization           | C10: Handling Concept Drift and
 Evolving Normality                                                                | Normal behavior may evolve over time; failure to adapt
 causes false alarms, while over-adaptation risks
 misclassifying anomalies     | [56], [57], [64]                    |
| Evaluation and
 Benchmarking
 Challenges | C11: Scarcity of Comprehensiv
 Benchmark Datasets                                                                  | Benchmark datasets are limited in diversity and detail                                                                                 | [37]–[39]                           |
| Evaluation and
 Benchmarking
 Challenges | C12: Limitations of Current
 Evaluation Metrics
 C13: Gap Between Offline                                          | ommon metrics often fail to reflect deployment
 performance.                                                                           | [8], [39], [41], [42]               |
| Evaluation and
 Benchmarking
 Challenges | C13: Gap Between Offline
 Evaluation and Deployment
 Performance                                                   | Real-time scenarios require new protocols for accurate
 assessment.                                                                    | [5], [41], [42]                     |
| Real-Time and
 Deployment
 Challenges    | 4: Real-Time Processing and
 Low Latency                                                                           | Timely detection is essential in safety-critical domains.                                                                              | [76]–[78]                           |
| Real-Time and
 Deployment
 Challenges    | Resource Constraints and
 Scalability
 Clibid hhldi                                                                | odels require significant computationa
 resources.
 hlhhld iiil b                                                                      | [8], [56], [79]                     |
| Real-Time and
 Deployment
 Challenges    | C16: Calibration and Thresholding
 (False Alarms vs. Misses)
 C17ChiFi                                             | g the right anomaly threshold is critical to balance
 false positives and false negatives.                                             | [80]–[82]                           |
| Adaptive Learning
 Challenges            | C17: Catastrophic Forgetting or
 Stability-Plasticity Dilemma
 C18: Efficient Label Utilization
 During Adaptation | odels may lose previously learned information whe
 updated with new data.                                                              | [57], [83], [84                     |
| Challenges                               | C18: Efficient Label Utilization
 During Adaptation                                                                | Labels are scarce in streaming settings                                                                                                | [43], [56], [57]                    |

must contend with background clutter, illumination changes, and camera motion that can obscure the relevant pattern. An emerging approach is to use other modalities such as pose, optical flows, object landmarks, etc. However, these approaches rely on accurate preprocessing steps. Additionally, detecting certain anomalies requires detailed visual queues that may be lost in higher levels of abstraction (e.g., detecting someone carrying a weapon would be more challenging).

## C. Robustness and Generalization Challenges

- 1) C7: Environmental Variations and Noise: VAD methods must operate in diverse real-world conditions that can affect their inputs. Models may face day/night cycles, various weather conditions, and lighting changes. These factors can introduce visual noise that are unrelated to anomalies but can confuse deep models. Another aspect is highly dynamic backgrounds that could lead to high false alarm rates when the model interprets normal background changes as abnormal. Robustness to these perturbations is crucial.

2) C8: Domain Shift and Cross-Scene Generalization: Related to C7 is the domain shift problem: an anomaly detection model trained in one setting often fails when deployed in a new setting. This is because deep models internalize the statistics of their training data's environment. Domain adaptation and generalization techniques are actively researched. This is a critical issue for scalability as well: a city-wide deployment across hundreds of street cameras would require per-camera calibration if the model cannot generalize.

3) C9: Open-Set Nature of Anomalies and Novelty: VAD is an open-set problem: a model can never see examples of all possible anomalies in training, since by definition anomalies encompass anything that deviates from normal, including novel events that have never occurred before. On the same note, capturing all normal behaviors is also not feasible. VAD must be prepared for the unforeseen. This translates to the "unknown unknowns" problem: an AI may handle known rare events but fail to recognize a truly odd hazard as an anomaly. The open-set challenge also complicates evaluation. A model could correctly detect all anomalies in a test set and still be unreliable in practice if a new kind of anomaly occurs.

## 4) C10: Handling Concept Drift and Evolving Normality:

Over time, what is considered normal or even anomalous may evolve. This phenomenon is known as concept drift. In a traffic monitoring scenario seasonal differences may cause normal behavior patterns to shift. In healthcare, a patient's baseline behavior might gradually change due to therapy or disease progression. If a model is not updated, it may raise false alarms on these evolving behaviors.

## D. Evaluation and Benchmarking Challenges

- 1) C11: Scarcity of Comprehensive Benchmark Datasets: Most current VAD models are trained and tested on a limited set of benchmarks. While useful for initial development, these datasets often lack diversity in scenes, environmental conditions, and anomaly categories.
- 2) C12: Limitations of Current Evaluation Metrics: The dominant metrics used in VAD fail to fully capture the realworld effectiveness of a model. These metrics often abstract away threshold selection and ignore the impact of false alarms. Additionally, metrics rarely account for operational concerns such as alert fatigue, latency, or the cost of misclassification.
- 3) C13: Gap Between Offline Evaluation and Deployment Performance: Many VAD methods are evaluated in offline settings using pre-recorded video clips. Offline evaluation may overstate model accuracy. Bridging the gap between offline benchmarks and online performance requires new evaluation protocols that account for temporal causality, resource constraints, and continuous learning needs.

## E. Real-Time and Deployment Challenges

- 1) C14: Real-Time Processing and Low Latency: For many applications, detecting anomalies promptly is crucial. For example, autonomous vehicles must detect and react to road anomalies within a very short time to avoid accidents. Such scenarios demand that deep learning models operate in real-time on video streams. Even if accuracy is high, a method that triggers an alert too late is often unacceptable in practice. Achieving real-time anomaly detection without sacrificing detection quality is an active challenge.
- 2) C15: Resource Constraints and Scalability: Deep learning models for video require significant memory and computation. In a real-world deployment like city-wide surveillance, running a deep anomaly detector on all feeds simultaneously is a massive scalability challenge. Likewise, an autonomous vehicle has a power and hardware budget. Thus, anomaly detection methods must be efficient in terms of computation, memory, and energy. Another aspect of scalability is handling long durations and continuous monitoring: a model might need to run 24/7. Storing and analyzing such long video sequences can be difficult. There is also a data management challenge: if anomalies are flagged often, how to store or review these events efficiently. Ensuring that a solution scales from a small benchmark to a deployment is a non-trivial jump.
- 3) C16: Calibration and Thresholding (False Alarms vs. Misses): Deploying an anomaly detector in the real world requires choosing how sensitive it should be. In other words, setting thresholds or decision criteria for what is flagged as anomalous. This leads to a classic precision-recall trade-off: a very sensitive system will catch nearly all true anomalies (high recall) but at the cost of many false alarms (low precision), whereas a strict system will raise fewer false alerts but might miss anomalies. Finding the right balance is extremely challenging and often application-specific.

## F. Adaptive Learning Challenges

1) C17: Catastrophic Forgetting or Stability-Plasticity Dilemma: Catastrophic forgetting is the tendency of models to overwrite previously learned knowledge when updated with new data. If a model is updated incrementally to learn from new scenes or behaviors, it may degrade in performance on previously seen data. This is critical in safety or surveillance settings, where remembering rare but significant events is essential. This challenge is closely related to the StabilityPlasticity Dilemma, which describes the trade-off between retaining existing knowledge (stability) and acquiring new knowledge (plasticity) without interference.

## 2) C18: Efficient Label Utilization During Adaptation:

Obtaining labels in a streaming setting is expensive and timeconsuming. Therefore, continual learning must proceed with minimal supervision. Designing models that can effectively leverage sparse and noisy labels, or self-supervise their adaptation process, is a key challenge.

## III. DEFINITIONS AND SOLUTIONS

VAD focuses on identifying patterns or events in video sequences that deviate significantly from expected or normal behavior [5]. As discussed in Section I, the complexity of VAD has led researchers to adopt varying levels of supervision (see Figure 6). This section identifies and discusses the definition and solutions within each supervision level. Table II summarizes the solutions and their weaknesses and strengths.

## A. Supervised VAD

Supervised VAD involves training models on labeled datasets where both normal and anomalous events are explicitly annotated. This approach is particularly effective in domains where anomalies are well-defined and annotated data is available, such as in healthcare. By learning from labeled examples, supervised methods can achieve high accuracy in detecting known types of anomalies. However, as outlined in Section II due to challenges such as Rarity of Anomalies and Class Imbalance, Limited and Difficult Labeling, Ambiguity in Defining Anomalies and Context Specificity, and Open-set nature of Anomalies and Novelty (challenges C1, C2, C3, and C9) supervised approaches exhibit limited applicability [5], [8], [35], [85], [86]. The main solution in this supervision level is treating VAD as a classification problem.

Supervised Classification (S1): The most common formulation of supervised VAD is as a classification task, where models are trained to distinguish between predefined normal and anomalous events. This setup leverages well-established classification algorithms to learn discriminative features.

## B. Weakly-Supervised VAD

To address the challenge of obtaining accurately labeled data for supervised solutions (challenges C2 and C3), weaklysupervised approaches offer greater flexibility. In these methods, labels may be incomplete, noisy, or ambiguous. Weaklysupervised solutions mostly take advantage of Multiple Instance Learning (MIL) and try to improve it for better efficacy.

Multiple Instance Learning (S2): MIL treats each video as a bag of instances, labeling it anomalous if at least one instance is abnormal (see Figure 2). During training, the model

Fig. 2. VAD formulated as a weakly supervised problem, commonly addressed using MIL (S2).

![Image](artifacts/image_000001_7309d0e041a83f6b8b5eac633fc3a663102e1188e4d2652e5c6b9cd5c577ca5f.png)

Fig. 3. Self/semi-supervised VAD achieved through reconstruction(S3) or prediction (S4). The top figure illustrates the training phase using only normal data, while the bottom shows the inference phase, where elevated loss indicates abnormal behavior.

![Image](artifacts/image_000002_2ef5f2761dc71a29d7ef805ff8f4f5f0605fe2564d5bd2eaec57ac773e74abe0.png)

learns to identify which instances within positive bags are anomalous, without needing fine-grained labels. One-stage MIL often focuses on the most prominent anomaly, risking missed detections of subtle instances, while two-stage selftraining methods use MIL to generate pseudo-labels iteratively, refining both the model and labels, enabling more robust and comprehensive detection of both obvious and subtle anomalies.

## C. Self/Semi-supervised VAD

Semi-supervised solutions bridge supervised and unsupervised learning paradigms by using only normal videos during training to learn the characteristics of normal behavior. Previous literature often classified these methods as unsupervised. However, recent works [54], [55] have reclassified them as semi-supervised due to the inherent supervision involved: normal and abnormal sequences are distinguished, and only normal sequences are utilized during training. This shift in terminology acknowledges the partial labeling and guidance provided, which differentiates these methods from truly unsupervised approaches. In general, most of these methods also fall under the self-supervised paradigm, where supervisory signals are derived from inherent characteristics of the normal data. Depending on the learning objective, these solutions can be categorized into four main groups: Reconstruction-based, Prediction-based, Jigsaw Puzzle, and Distribution Estimation.

Reconstruction-based (S3): This strategy employs autoencoders to reconstruct normal data; anomalies are indicated by high reconstruction loss when the model fails to reconstruct anomalous snippets accurately, as seen in Figure 3.

Prediction-based (S4): In these approaches, models are trained to predict the future normal behavior, with anomalies identified through higher prediction loss on abnormal sequences, as seen in Figure 3.

Jigsaw Puzzle (S5): A supervisory signal is generated by formulating a jigsaw puzzle task, which may be spatial, temporal, or a combination of both (see Figure 4). The model is trained exclusively on normal data, learning to reassemble shuffled video segments. During inference, its ability to correctly reconstruct these sequences is used as a measure for computing anomaly scores.

Distribution Estimation (S6): This category employs either non-deep learning or deep learning methods to model the distribution of normal samples during training. At inference, instances with low likelihood are identified as anomalies.

## D. Unsupervised

In unsupervised training, no labels are available to distinguish between normal and anomalous instances. However, the literature frequently misclassifies certain self-supervised or semi-supervised approaches as unsupervised. A critical observation is that many of these methods are trained exclusively on normal data. This implicitly introduces label information, violating the core principle of unsupervised learning [54]. Consequently, such models should not be considered unsupervised. The degree of supervision must be evaluated not only based on the methodology but also in relation to the informational content embedded in the training data. Despite its fundamental nature, fully unsupervised anomaly detection remains relatively underexplored compared to its self-supervised and semisupervised counterparts, indicating a significant opportunity for advancement and application in real-world scenarios.

Truly unsupervised methods operate without any access to ground-truth labels. These approaches aim to exploit the normality advantage; the observation that anomalies represent rare and irregular events, whereas the majority of the data corresponds to normal behavior [87], [88]. The core strategy behind these methods is to leverage this statistical imbalance: given that normal samples dominate the dataset, the global structure and distributional trends of the data are expected to reflect normal characteristics. Unsupervised models are therefore trained to capture these prevailing patterns, under the assumption that deviations from the learned representation will correspond to anomalous instances.

Clustering (S7): Clustering methods assume that normal data form dense clusters in feature space, while outliers in low-density regions are potential anomalies, as illustrated in Figure 5. Approaches range from classical algorithms like kmeans to deep clustering methods that jointly learn features and clusters. Despite their effectiveness, clustering methods face challenges such as sensitivity to hyperparameters, such as the number of clusters, and reliance on clear structural differences between normal and anomalous data.

Pseudo-label Induction (S8): This strategy leverages the normality assumption: normal data dominate the input distribution. While conceptually related to self or semi-supervised approaches, a key distinction is that the training data includes unknown anomalies. These methods use reconstruction errors or prediction inconsistencies to assign pseudo-labels, guiding anomaly filtering or classifier training. As they rely on selfgenerated signals without ground truth, they are considered unsupervised self-supervised approaches. However, unreliable pseudo-labels and feedback loops can undermine robustness and generalizability, especially in noisy or complex data.

Fig. 4. Self/semi-supervised VAD achieved through jigsaw puzzle task (S5). The puzzle can be spatial, temporal, or a combination. The left figure illustrates the training phase using only normal data, while the right shows the inference phase, where wrong permutation prediction indicates abnormal behavior.

![Image](artifacts/image_000003_03cef2c174968423b60444e0ddf2b309c4d2cbc654764443239bbde5401b4d00.png)

Fig. 5. Unsupervised anomaly detection through clustering.

![Image](artifacts/image_000004_82bf5e80201e99669578140533bda5493526d51a487e01e615aa8d5016da4c65.png)

## IV. ADAPTIVE LEARNING IN VAD

Fig. 6. Percentage distribution of supervision levels within each domain.

![Image](artifacts/image_000005_863851e37829b31602d8363fa2cbaa4410fe02e99a4e4fda5679bb7d21af7a7e.png)

As discussed in Section I and Section II, VAD is a dynamic and complex problem, ever evolving and heavily affected by spatio-temporal changes. This includes but is not limited to Environmental Variations and Noise, Domain Shift, OpenSet Nature, Concept Drift, and Calibration and Thresholding

(challenges C7, C8, C9, C10, and C16). To address these challenges, adaptive learning methods such as meta-learning, online learning, continual learning, and active learning have become essential [89]–[92]. In this survey, the term "adaptive

TABLE II OVERALL CLASSIFICATION OF VAD SOLUTIONS .

| Supervision
 Level           | Solution                                                                                | Definition                                                                              | Main Strength                                                                                     | Main Limitation                                                                                               |
|------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Supervised                   | S1: Supervised
 Classification                                                          | Frames VAD as a classification
 task using labeled datasets.                            | High accuracy and reliability in
 detecting predefined, labeled
 anomalies.                       | Supervised                                                                                                    |
| WeaklySupervised            | S2: Multiple Instance
 Learning                                                         | Labels video bags; identifies
 anomalous instances.                                     | Can handle weak labels where
 only bag-level annotation is
 provided, reducing labeling
 efforts. | WeaklySupervised                                                                                             |
| Supervise                    | S3: Reconstruction                                                                      | Uses autoencoders to reconstruct
 ormal data; anomalies have high
 reconstruction loss. | Effective for capturing the
 structure of normal behaviors                                        | Struggles with generalization
 when normal patterns exhibit
 high variability.
 Suffers in scenarios with non |
| pervised                     | Predicts future behavio
 prediction loss indic
 anomalies.                              | Predicts future behavior; high
 prediction loss indicates
 anomalies.                   | Effective for capturing the
 structure of normal behaviors                                        | Struggles with generalization
 when normal patterns exhibit
 high variability.
 Suffers in scenarios with non |
| S5: Jigsaw Puzzle            | Challenges models to reassembl
 shuffled video segments.                                | ubtle
 Complexity of s
 permutations in 
 puzzles, affecting r
 VAD.                    | Effective for capturing the
 structure of normal behaviors                                        | Complexity of solving
 permutations in jigsaw
 puzzles, affecting real-time
 VAD.                             |
| S6: Distribution
 Estimation | Uses generative models or
 statistical methods to learn normal
 behavior distributions. | tional properties
 behaviors. Sen                                                       | Effective for capturing the
 structure of normal behaviors                                        | Sensitive to noise.                                                                                           |
| Unsupervised                 | S7: Clustering                                                                          | No reliance on labeled dat
 simple implementation                                       | No reliance on labeled data and
 simple implementation.                                           | Limited generalizability,
 sensitive to hyperparameters                                                       |
|                              | S8: Pseudo-Label
 Induction                                                             | verage error magnitude to do
 pseudo-labeling for filtering
 anomalies.                 | No reliance on labeled data and
 simple implementation.                                           | do
 No reliance on labeled data.
 Pseudo labels are uncertain
 and can potentially reinforce
 false patterns. |

learning" encompasses a range of general adaptation methods. These techniques enable models to update and adjust to new data, trying to manage the aforementioned challenges.

Meta-learning, also known as "learning to learn," [93] focuses on designing models capable of rapidly adapting to new tasks by leveraging knowledge acquired from previous tasks. This approach involves training across a variety of tasks to develop a general learning strategy, enabling the model to perform effectively on novel tasks with minimal data, which is particularly useful for solving the domain shift problems in VAD models. One significant weakness of meta-learning, particularly in real-time VAD, is the high computational expense associated with training across multiple tasks. However, integrating meta-learning with few-shot training methods can help mitigate this issue by enabling the model to learn from a limited number of examples, thereby reducing the computational burden while maintaining adaptability and performance. [94] introduces a meta-learning framework using the ModelAgnostic Meta-Learning (MAML) [95] algorithm to enhance semi/self-supervised anomaly detection in surveillance videos. This approach involves training the model on various scenes, creating tasks that simulate few-shot learning scenarios.

Online Learning is a paradigm where the model is updated incrementally as it receives new data points [96], [97]. This approach allows the model to adapt continuously to new information. Online learning is particularly advantageous when dealing with large datasets or streaming data, as it can handle data efficiently without requiring access to the entire dataset simultaneously. Online learning has been explored for anomaly detection on other types of data, such as time series [90], text [98], and medical images [99]. In VAD, Yao et al. [100] introduced a framework optimized for real-world deployment, integrating inference and training in a pipeline to enhance public safety applications. While effective under conditions of minimal distributional shift, online learning faces notable limitations. These include susceptibility to noisy or unrepresentative data (challenges C1 and C7), as well as challenges such as the stability-plasticity dilemma and catastrophic forgetting (challenge C17), where frequent updates may overwrite prior knowledge.

Continual Learning is a strategy in machine learning where a model is designed to continually acquire, fine-tune, and retain knowledge from a stream of data over an extended period. This approach addresses the challenge of catastrophic forgetting (challenge C17), where learning new information can lead to a loss of previously acquired knowledge [101], [102]. This enables models to adapt to new tasks and changes in data distribution without sacrificing performance on previously learned tasks, making it particularly valuable in dynamic environments where the data evolves over time. Continual learning encounters challenges related to managing the high volume of streaming data and maintaining the efficiency of continuous model training. That is why most of the works in this area move toward few-shot learning to be able to handle the complexity of the training process while making real-time decisions. [56] proposed a two-step method for anomaly detection using deep learning-based feature extractors combined with kNN and a memory module, enhanced by two continual learning approaches. The first approach involves exact k-Nearest Neighbor kNN distance computation, effective for incrementally learning nominal behaviors when the training data size is manageable, updating the memory module with kNN distances from each training split. To address the computational expense as the training set grows, the second approach employs a fully connected deep neural network (kDNN) to estimate kNN distances, ensuring scalability and efficiency.

Active Learning [103] is a technique where the algorithm selectively queries the user (or domain experts) to label new data points to improve the learning efficiency and model performance [104], [105]. In scenarios where labeling data is costly or time-consuming, active learning is particularly valuable because it allows the model to focus on acquiring labels for the most informative data points. This is achieved through various strategies that prioritize data points based on criteria such as uncertainty, representativeness, or expected model change [105]. By enabling the model to query the most useful data points for annotation, active learning reduces the need for large pre-labeled data and enhances the model's ability to generalize from fewer labeled instances (challenges C1, C2, and C18). Incorporating human feedback on selected samples within an active learning framework establishes a fewshot learning paradigm that improves the efficacy of anomaly detection systems and the efficiency of training the model. A significant challenge associated with this technique is the requisite involvement of a human or domain expert (challenges C2 and C18). This requirement can introduce complexities related to scalability and efficiency, as the continuous need for expert input can limit the speed and autonomy of the learning process. [52] proposes an active learning framework using YOLO v3 [106] and Flownet 2 [107] for feature extraction and kNN for anomaly detection. The model constructs a statistical baseline of normal behaviors using kNN distances and continually updates it with new nominal data. Anomalies trigger human feedback for labeling, which categorizes this work as an active learning framework rather than continual learning, as described in the original paper. Several other works propose a more advanced method for selecting the data queries. [108] proposes an adaptive weighting scheme for dynamically selecting between various criteria such as the likelihood criterion, which selects samples with low likelihood according to the current model to discover new classes, and the uncertainty criterion, which selects samples that cause the most disagreement among committee members to refine the decision boundary. [109] utilizes a Bayesian nonparametric model, specifically the Pitman-Yor Process (PYP) for managing imbalanced class distributions (challenge C1) and models probabilities for both known and unknown classes.

## V. HUMAN-CENTRIC VAD

## A. Healthcare

In healthcare VAD, the goal is to detect deviations in physiological or behavioral patterns that may signal disease, injury, or other medical conditions, enabling early diagnosis and intervention to improve outcomes and reduce costs. These systems might process various data types, but in this work,

TABLE III REVIEWED WORKS IN HEALTHCARE: ALL STUDIES EMPLOY SUPERVISED LEARNING; * DENOTES STUDIES THAT EVALUATE MULTIPLE ARCHITECTURES .

| Task                   | Approach      | Architecture                     | Distinct Characteristics / Novel Contri                                                                                                                                                                 | Modalit                      |
|------------------------|---------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| Task 
 Fall
 Detection | [110]         | CNN, LSTM                        | Performs person detection and contour-based feature extraction, follo
 by attentionguided LSTM                                                                                                          | RGB                          |
| Task 
 Fall
 Detection | [111]         | CNN                              | Mitigates feature loss through multi-task learning and leverages la
 features for decision-making                                                                                                       | RGB                          |
| Task 
 Fall
 Detection | [112]         | MLP                              | Proposes enhanced optical dynamic flow for improved temporal motion
 estimation in fall scenarios 
 Ubd ffll idbd bd                                                                                    | Optical Flow                 |
| Task 
 Fall
 Detection | [113]         | Heuristic Rule-based
 Model      | Uses pose-based features to compute fall index based on body posture
 changes 
 Ctttitl h f hd lih                                                                                                      | Pose                         |
| Task 
 Fall
 Detection | [114]         | GCN                              | Constructs a spatiotemporal graph of human poses and applies grap
 convolution                                                                                                                          | Pose                         |
| Task 
 Fall
 Detection | [46]          | Random Forest, MLP 
 CNNLogistic | Divides falls into dynamic/static states; uses fusion of vision-based da
 dlbddiih id dld li                                                                                                            | RGB, Pose                    |
| Task 
 Fall
 Detection | [115]         | CNN, Logistic
 Regression        | ls body dynamics with an inverted pendulum and analyzes motio
 stability to extract features                                                                                                            | RGB, Pose                    |
| Task 
 Fall
 Detection | [116]         | CNN                              | A multi-stream CNN where each stream processes different features                                                                                                                                       | RGB, Depth, Optical
 Flow    |
| Parkinson’s
 Detection | [117]         | Deep Residual
 Network           | multimodal system using facial features and expression-specific act
 for effective detection 
 Alkd fil iididiff                                                                                        | RGB (Facial Video)           |
| Parkinson’s
 Detection | [118]         | CNN, SVM                         | yzes evoked facial expressions using domain adaptation from fac
 recognition 
 gait energy images to classify Parkinson’s gait leveraging onecla                                                        | RGB (Facial Video            |
| Parkinson’s
 Detection | [118]         | CNN, SVM                         | gait energy images to classify Parkinson’s gait leveraging one-clas
 SVM 
 tit dftibtti2D tti3D i                                                                                                       | GB (Facial Vid               |
| Parkinson’s
 Detection | [119]         | *                                | icts gait dysfunction by extracting 2D poses, reconstructing 3D ga
 multiviews, and analyzing features using classical and deep learnin
 models                                                         | Gait                         |
| Parkinson’s
 Detection | [120]         | Random Fores                     | yzes stride variability and cadence using pose-based features for
 effective detection                                                                                                                  | Pose                         |
| Parkinson’s
 Detection | [48]          | *                                | es Parkinson’s symptoms via jitter and amplitude of small muscle
 groups in face videos                                                                                                                 | Facial Landmarks             |
| Parkinson’s
 Detection | [121]         | *                                | ms remote assessment using webcam video by extracting hand
 landmarks                                                                                                                                   | Hand Landmarks               |
| Parkinson’s
 Detection | [122]         | CNN, Random Forest               | eye-tracking and gait data using covariance descriptors for
 Parkinson progression quantification                                                                                                       | RGB (Eye Video),
 Gait       |
| Autism
 Detection      | [123]         | 3D CNN, LSTM                     | oses spatial attentional bilinear pooling to capture fine-grained
 atial features and dynamic attention on discriminative regions 
 tegrates phototaking and imageviewing modalities through            | RGB                          |
| Autism
 Detection      | [124]         | CNN, LSTM                        | Integrates phototaking and imageviewing modalities through
 ti-modal knowledge distillation, enabling accurate detection using
 temporal and attentional behavioral features                            | RGB                          |
| Autism
 Detection      | [125]         | CNN, SVM                         | Analyzes attention pattern differences using discriminative image
 selection and fixation maps, followed by linear SVM classification 
 Extracts visual and temporal features from gaze scanpaths using | RGB (Eye Video)              |
| Autism
 Detection      | [126]         | CNN, LSTM                        | Extracts visual and temporal features from gaze scanpaths using
 saliency-guided patch extraction for sequence-based prediction                                                                         | Scanpath                     |
| Autism
 Detection      | [127]         | 3D-CNN                           | izes 3D-CNN for spatiotemporal analysis, focusing on a
 recognition to detect symptoms                                                                                                                  | RGB, Optical Flow            |
| Autism
 Detection      | [128]         | CNN, MLP                         | Processes facial expressions for autism screening                                                                                                                                                       | RGB, Facial                  |
| Autism
 Detection      | [129]         | LSTM                             | es on posture and movement data in social interactions for detection                                                                                                                                    | Pose                         |
| Seizure
 Detection     | [130]         | CNNLSTM                          | Analyzes spatial vs. spatiotemporal features for detection, showing the
 latter performs better                                                                                                         | RGB                          |
| Seizure
 Detection     | [131]         | Transformer                      | pplies BART-inspired self-supervised training on hospital videos to
 learn contextfollowed by classification for seizure detection                                                                      | RGB                          |
| Seizure
 Detection     | [132]         | ansforme                         | pplies BART-inspired self-supervised training on hospital videos t
 learn contextfollowed by classification for seizure detection                                                                       | RGB                          |
| Seizure
 Detection     | [132]         | *                                | learn context, followed by classification for seizure detection 
 Emotion detection used as a feature extractor 
 Reconstructs 3D facial geometry to capture mouth and cheek motions,                   | RGB (Facial Video)           |
| Seizure
 Detection     | [134]         | CNN                              | p, 
 temporal dynamics for seizure classification
 Transforms EEG into second-order Poincare plots and uses pre-trained ´
 CNNtlifit                                                                    | RGB (EEG Vid)                |
| Seizure
 Detection     | [135]         | SVM                              | Applies dimensionality reduction techniques (PCA and ICA), and defines
 handcrafted features for a SVM classifier                                                                                       | Optical Flow                 |
| Seizure
 Detection     | SETR [49]     | Transformer                      | Uses pretrained networks for spatial features, a transformer for temporal
 modeling, and Progressive Knowledge Distillation for early detection 
 iiihlii                                               | Optical Flow                 |
| Seizure
 Detection     | [136]         | CNN                              | Generates a compact image representation capturing the location variance
 and periodicity of semiology                                                                                                  | RGB, Optical Flow            |
| Seizure
 Detection     | [137],
 [138] | GCN, TCN                         | A multistream framework leveraging GCN, spatio-temporal feature
 extraction, and late fusion                                                                                                            | RGB, Pose, Facial
 Ldk       |
| Seizure
 Detection     | []
 [138]     | GCN, TCN                         | gg , pp
 extraction, and late fusion                                                                                                                                                                    | RGB, Pose, Facial
 Landmarks |

we focused on methods that use video as their primary data. As shown in Figure 1, most approaches adopt a supervised learning paradigm, reflecting the domain's need for precise, reliable detection. Research primarily targets events with strong visual cues, such as falls, Parkinson's episodes, autistic behaviors, and seizures, which exhibit distinctive motion or posture patterns amenable to visual analysis (see Table III).

1) Fall Detection: Fall detection is a key task in healthcarerelated video analysis due to its distinct visual patterns and practical significance. Early methods typically use RGB video and leverage pre-trained models, applying object detection and temporal modeling to track human motion. For instance, [110] uses an LSTM to distinguish fall-like behaviors over time, while [111] proposes a two-stage approach with a convolutional autoencoder for feature extraction, followed by a lightweight classifier for final prediction.

To address the limitations of RGB-only approaches, particularly under challenging conditions such as poor lighting, occlusions, or background clutter, recent works have incorporated additional modalities to improve fall detection

performance. Optical flow captures pixel-level motion between frames, offering a richer representation of dynamic events; for instance, [112] uses optical flow with a fine-tuned VGG16 network [139] to enhance motion-specific feature learning. Human pose estimation further improves robustness by abstracting subjects into skeletal representations, which are less sensitive to visual noise. Pose-based features such as centroid velocity and rotational energy have been applied using both deep learning and traditional classifiers, including logistic regression [115], and hybrid models like the Multi Layer Perceptron (MLP) combined with random forest in [46]. Advancing beyond handcrafted descriptors, [114] introduces a spatiotemporal graph convolutional network (ST-GCN) for end-to-end learning of pose dynamics. To further enhance robustness and capture complementary information, some studies combine multiple modalities such as RGB, depth maps, optical flow, and motion history images, processed through specialized network branches [116].

2) Parkinson Detection: Parkinson's disease (PD) exhibits both motor and non-motor symptoms, with motor manifestations such as tremor, rigidity, bradykinesia, postural instability, and shuffling gait being the most visually detectable and thus well-suited for computer vision analysis. Leveraging this visual accessibility, recent research has focused on facial and body movement analysis to identify Parkinsonian signs. A key facial symptom, hypomimia (reduced expressiveness) has been widely studied. For instance, [48] applies facial landmark detection to extract handcrafted features classified with traditional algorithms, while [117] enhances facial analysis through segmentation and hybrid learning strategies. More recent endto-end approaches, such as [118], repurpose pretrained face recognition models via transfer learning to detect PD and assess motor impairment severity using multiple Support Vector Machine (SVM) classifiers.

Another line of research focuses on gait and pose-based analysis, targeting motor irregularities such as bradykinesia (slowness and reduced movement amplitude) common in PD. For example, [121] uses hand keypoint trajectories during motor tasks, classifying temporal patterns with conventional models like logistic regression and random forests. Other studies analyze full-body motion through silhouettes or skeletal poses; [47] creates Gait Energy Images (GEIs), while [120] applies pose estimation followed by SVM classification. A more advanced approach by [119] combines multiview RGB video with 3D skeletal reconstruction and deep models, including multi-scale residual networks, achieving strong generalization. Some studies have sought to combine multiple modalities, such as facial and body movement cues, to enhance detection robustness. For instance, [122] proposes a multimodal framework that integrates facial expressions and skeletal motion features, aiming to capture complementary signals associated with Parkinsonian motor deficits.

3) Autism Detection: Autism Spectrum Disorder (ASD) is a common neurodevelopmental condition in children, marked by social communication deficits and atypical attention patterns. Clinical assessment relies on repeated, time-intensive behavioral evaluations by trained professionals, which are prone to subjective variability. As a result, developing automated, objective tools for ASD detection is critical to enable early, consistent, and scalable diagnosis.

One research direction leverages eye gaze patterns as behavioral biomarkers for ASD, given their link to impaired social engagement and disruptions in the social brain network. Jiang et al. [125] used VGG-16 to analyze fixation difference maps and classified visual attention features with a linear SVM. Chen and Zhao [124] combined ResNet-50 [140] with LSTM layers to model spatial-temporal gaze dynamics. Tao et al. [126] proposed SP-ASDNet, using saliency maps from neurotypical individuals to guide patch selection, followed by a CNN-LSTM network to detect deviations indicative of ASD.

Beyond gaze, many studies focus on general behavioral patterns, especially stereotypical behaviors like clapping, arm flapping, and repetitive movements, common indicators in ASD diagnosis. Ali et al. [127] use 3D CNNs to detect such actions, supporting clinical assessments without providing a final diagnosis. Wu et al. [128] offer a more integrated pipeline, combining deep models on RGB and facial landmarks with statistical features (e.g., behavior frequency and duration), fed into a neural network for classification, linking low-level behavior detection with high-level diagnostic inference.

A more recent, data-driven approach eliminates manual feature engineering by end-to-end deep learning models that learn discriminative patterns directly from video. Sun et al. [123] combine CNNs with LSTMs to extract spatial-temporal features from pixel data, while Kojovic et al. [129] use a similar architecture with human pose inputs, offering a more abstract and potentially robust representation.

Together, these studies form a continuum from explicit behavior modeling to implicit feature learning, highlighting the progression toward more generalizable and efficient systems. Each category of methods, whether based on gaze analysis, stereotyped motor behavior, or end-to-end learning, addresses different aspects of the complex behavioral phenotype associated with ASD, and collectively, they underscore the potential of machine learning in revolutionizing autism diagnosis.

4) Seizure Detection: Seizure detection has traditionally relied on Electroencephalography (EEG), often paired with video (VEEG) to link motor behaviors with brain activity. Some works, such as [134], convert EEG data into visual forms like Poincare plots for classification via pre-trained ´ ´ CNNs. While effective, EEG remains intrusive and impractical for long-term or ambulatory use. As a result, recent efforts have focused on video-only systems that analyze visible cues such as facial expressions and body movements, offering noninvasive, scalable, and more comfortable alternatives.

Building on this shift, recent work has explored methods focusing on facial features and expressions, particularly facial semiology (e.g., involuntary movements). Pothula et al. [132] use standard facial recognition pipelines to extract features for classification, while Ahmedt-Aristizabal et al. [133] enhance this by modeling 3D facial dynamics, especially mouth motion, using LSTM networks.

Another research direction focuses on full-body movement, which is more pronounced in generalized seizures. Yang et al. [130] use CNNs and LSTMs to capture spatial and temporal motion features. More recent work, such as Hou et al. [131],

introduces transformer-based models with BART-style selfsupervised pretraining, enabling effective seizure classification with reduced dependence on large labeled datasets.

To address privacy concerns in video-based seizure monitoring, recent studies use de-identified features such as optical flow, which captures motion without revealing identity. Garc¸ao et al. [135] apply dimensionality reduction and SVMs ˜ ˜ to optical flow, while Mehta et al. [49] propose a CNNtransformer hybrid with Progressive Knowledge Distillation for early prediction. Complementing this, multimodal fusion strategies have been explored to improve detection robustness. Ahmedt-Aristizabal et al. [136] integrate facial and hand movements to create compact semiology descriptors, while Hou et al. [137], [138] fuse RGB, optical flow, body pose, and facial landmarks via multi-branch networks to produce richer representations for seizure classification.

## B. Public Safety

In public safety, video anomaly detection focuses on identifying risky behaviors like violence or rule violations by analyzing external cues. Pixel-based methods capture rich context but are sensitive to environment changes and privacy issues, while pose-based approaches improve robustness and privacy at the cost of visual detail. Some studies combine both in multimodal frameworks. Real-time applications demand efficient trade-offs between accuracy, privacy, and speed.

1) Pixel-Based Methods: In the context of public safety, pixel-based methods for VAD continue to play a central role due to their ability to capture fine-grained visual details, including both environmental context and object appearance. Table IV summarizes these methods. Some works pursue taskspecific anomaly detection, focusing on particular threats such as shoplifting [50], [141], [142], weapon detection [143], [144], [156], or vandalism [145]. While these methods offer high precision for well-defined scenarios, their generalizability remains limited, which motivates the mainstream research direction in VAD: detecting a broad range of anomalies without pre-defining their nature.

Recent progress in weakly supervised VAD has shown that it is possible to achieve fine-grained temporal localization using only video-level labels. A prominent direction in this field involves pseudo-label refinement: Tian et al. [154] propose a two-stage strategy using a multi-head classifier with diversity loss and Monte Carlo Dropout-based uncertainty filtering to generate high-quality pseudo labels. Similarly, Wang et al. [152] introduce ARMS, a multi-phase training framework that incrementally increases the assumed ratio of abnormal segments to progressively discover harder anomalies, supported by temporal convolution and attention. Complementary to these efforts, RTFM [32] avoids over-reliance on classifier outputs by focusing on feature magnitudes, selecting topk high-magnitude snippets to separate normal and abnormal segments using a multi-scale temporal architecture. In parallel, vision-language models have emerged as powerful tools for semantic alignment: Li et al. [155] utilize CLIP-based featuretext alignment combined with temporal context learning, while An et al. [153] adopt ViLBERT features in an MIL framework for snippet-level classification from coarse labels.

In self-supervised learning, models define proxy tasks to learn representations of normal behavior without requiring labeled anomalies, as mentioned in Section III. Among these, reconstruction-based methods have long been popular. To enhance reconstruction accuracy and enforce better anomaly separation, adversarial training has been widely adopted. For instance, Yang et al. [146] use a discriminator to distinguish between real and reconstructed patches, pushing the generator (autoencoder) to reconstruct more accurately. Chen et al. [147] instead use the discriminator to differentiate between real reconstruction error maps and synthetic noise, penalizing abnormality through structural deviations. In another approach, Georgescu et al. [150] use irrelevant pseudo-anomalies (e.g., flowers, anime images) to train a discriminator to separate pseudo-abnormal and normal samples, encouraging the generator to focus specifically on human behavioral features.

Prediction-based models have also evolved to integrate optical flow for more accurate future frame prediction. Luo et al. [151] replace basic MSE loss with a combination of flow, intensity, and gradient-based losses, alongside adversarial training for sharper predictions. Huang et al. [53] employ separate encoders for appearance and flow, feeding both into a unified decoder with skip connections and memory modules to compare current behavior with learned normal prototypes for better suppression of anomalies.

Other self-supervised tasks, such as jigsaw puzzle-based learning, aim to improve generalization by encouraging spatiotemporal reasoning. Wang et al. [149] decouple spatial and temporal dimensions to form dual puzzles, solved via a 3D CNN trained to predict permutations learning both visual structure and motion patterns. Further extending generalization, some methods employ multiple proxy tasks. Georgescu et al. [64] use a suite of four self-supervised tasks: arrowof-time prediction, motion shuffling, irregularity localization, and knowledge distillation, while its successor, SSMTL++ [148], adds jigsaw puzzles and adversarial pseudo-anomalies for broader robustness. Beyond reconstruction and prediction, Doshi et al. [52] use deep learning for feature extraction and statistical modeling to estimate the distribution of normal data, enabling adaptive decision-making through continual learning. Trained solely on normal data, the approach falls under semisupervised learning and focuses on dynamic thresholds in evolving environments.

While these approaches reduce dependence on labeled data, even self-supervised methods often assume that training videos are purely normal. Recent research aims to relax this assumption. ESSL [55] builds on puzzle-based learning but incorporates a self-selective module to identify and exclude suspected anomalies during training, enabling learning from mixed datasets. Similarly, Zaheer et al. [54] propose a Generative Cooperative Learning framework, where a generator reconstructs input features and a discriminator classifies them as normal or anomalous using pseudo-labels derived from reconstruction errors. A negative learning strategy intentionally trains the generator to reconstruct anomalous samples poorly, reinforcing clear distinctions between normal and abnormal patterns, achieving truly unsupervised anomaly detection.

TABLE IV

OVERVIEW OF PIXEL -BASED APPROACHES IN VAD FOR PUBLIC SAFETY. * DENOTES THAT MULTIPLE ALTERNATIVE ARCHITECTURES HAVE BEEN USED .

| Approach     | Supervision          | Strategy    | rchitecture    | Distinct Characteristics / Novel Cont                                                                                                                                                                      | Modalit     |
|--------------|----------------------|-------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| [50]         | Supervised           | S1          | CNN, RNN       | Using CNN as spatial feature extractor and RNN fo
 temporal pattern detection and final classification                                                                                                     | Pixel       |
| [141         | Supervised           | S1          | 3D CNN         | pp
 Use 3D CNN for simultaneous spatiotempo                                                                                                                                                                | Pixe        |
| [142         | Supervised           | S1          | CNN, LSTM      | Uses Inception V3 blocks and LSTM for feature extract                                                                                                                                                      | Pixel       |
| ADOS [143]   | Supervised           | S1          | CNN            | Minimizes multi-object detection errors by segmenting
 frames and applying a saliency-aware classification                                                                                                 | Pixel       |
| [51]         | Supervised           | S1          | *              | o-stage gun detection using fine-tuned spatial classi
 and temporal sequence models                                                                                                                        | Pixel       |
| [144]        | Supervised           | S1          | CNN            | Uses off-the-shelf object detectors and reduces fals
 positives by incorporating confusion classes 
 CNNLSTM deep learning model that combines spa                                                         | Pixel       |
| [145]        | Supervised           | S1          | CNN, LSTM      | CNNLSTM deep learning model that combines spatia
 feature extraction from convolutional layers with tempor
 sequence modeling from LSTM                                                                    | Pixel       |
| [146]        | Self/Semi-supervised | S4          | CNN            | y 
 reconstruction and object-focused scoring based on
 likelihood, position, and confidence
 Uses noisemodulated adversarial learningwhere a                                                              | Pixel       |
| NM-GAN [147] | Self/Semi-supervised | S4          | CNN            | Uses noisemodulated adversarial learning, where a
 discriminator trained on noise-injected reconstruction erro
 distinguishes normal from anomalous patterns                                               | Pixel       |
| [64]         | Self/Semi-supervised | S5, S6      | CNN, 3D CNN    | Defining multiple tasksarrow of time prediction, motio
 shuffling, irregularity prediction (viewed as various jigsaw
 puzzles), and knowledge distillation                                                 | Pixel       |
| [148]        | Self/Semi-supervised | S5, S6      | CNN, 3D CNN    | Adds adversarial pseudo anomalies, segmentation, jigsaw
 pose estimation, and inpainting to multi-task training 
 Dld til d tl jild                                                                        | Pixel       |
| [149]        | Self/Semi-supervised | S6          | 3D CNN         | Decoupled spatial and temporal jigsaw puzzles and
 employed a multi-label paradigm for more accurate VAD 
 Udbl ltidthtd                                                                                   | Pixel       |
| [150]        | Self/Semi-supervised | S4          | CNN            | Uses pseudo-abnormal examples to guide the autoencode
 and binary classifiers for each branch 
 A dldildddl tht li                                                                                         | Pixel, Flow |
| [53]         | Self/Semi-supervised | S5          | CNN            | A dualencoder singledecoder model that aligns appearan
 and flow features and uses memory of normal prototypes 
 enhance detection accuracy                                                                | Pixel, Flow |
| [151]        | Self/Semi-supervised | S5          | CNN            | Future prediction is guided by flow, intensity, and gradie
 losses, with a discriminator improving frame realism
 Combines flow and object detections to form feature vec                                  |             |
| ARMS [152]   | Weakly-supervised    | S2          | CNN            | Trained through bootstrapped pseudo labeling, hard anoma
 mining, and adaptive self-training with dynamic abnorma
 ratios to capture both easy and subtle anomalies
 Abl ihhihfid                          | Pixel       |
| RTFM [32]    | Weakly-supervised    | S2          | CNN            | Assumes abnormal snippets have higher feature magnitude
 selects top-k segments per video to maximize
 abnormal-normal separation                                                                          | Pixel       |
| [153]        | Weakly-supervised    | S2          | Transformer    | pg
 followed by a fully connected network trained with a
 soft-margin ranking loss on mean anomaly scores of
 positive and negative bags                                                                   | Pixel, Flow |
| [154]        | Weakly-supervised    | S2          | Transformer    | mproves pseudo labels through completeness modeling and
 diversity-enhanced multi-head classification, followed by
 uncertainty-aware self-training that selects reliable clips
 using Monte Carlo Dropout | Pixel, Flow |
| TPWNG [155]  | Weakly-supervised    | S2          | Transformer    | Uses CLIP for pseudolabeling, then trains a classifier wit
 a Temporal Context Self-Adaptive Learning module that
 adjusts attention spans based on event duration                                         | Pixel, Flow |
| ESSL [55]    | Unsupervised         | S8          | 3D CNN         | Extends the jigsaw puzzle concept with a self-selective
 module to filter potential anomalies, enabling truly
 unsupervised training                                                                       | Pixe        |
| [54]         |                      | S8          | 3D CNN         | xtends the jigsaw puzzle concept with a selfselective
 module to filter potential anomalies, enabling truly
 unsupervised training                                                                         | Pixe        |

2) Pose-Based Methods: Pose-based VAD has emerged as a powerful alternative to appearance-based methods, particularly in applications where privacy, robustness to environmental variation, and focus on human motion are essential (see Table V for summary). The dominant paradigm in this area is semi-supervised or self-supervised learning, where models aim to learn the regular patterns of human skeletal motion using only normal data. A central challenge lies in capturing the complexity of human movement while ensuring effective generalization to unseen abnormal patterns. To this end, researchers have explored increasingly sophisticated architectures that improve reconstruction or prediction quality by modeling the temporal and spatial dynamics of the human skeleton. Given the inherent graph structure of the human pose, where joints are nodes and limbs form edges, many works [42], [69], [158], [159], [161], [163]–[166], [169] naturally adopt graph-based architectures, particularly graph convolutional networks (GCNs), to model both temporal sequences and body structure.

Traditional reconstruction frameworks are extended by integrating more powerful sequence modeling mechanisms, such as transformers, which excel at capturing long-range dependencies. For instance, Yu et al. [70] propose a tokenization scheme based on the first-order difference between pose frames and introduce a motion prior derived from training statistics to explicitly model the distribution of joint displace-

ments, enhancing anomaly detection sensitivity.

A notable trend in recent years is the combination of reconstruction-based learning with distribution modeling. Several works [163], [164], [169] adopt a two-stage framework: first, training an autoencoder on normal data and then performing latent space clustering at test time to detect anomalies as outliers. Jain et al. [162] utilize a variational autoencoder (VAE) to impose a probabilistic structure on the latent space, enabling more principled distribution estimation. Extending this idea further, Hirschorn et al. [69] propose a purely probabilistic model using normalizing flows, where input pose sequences are transformed into a standard Gaussian distribution, and anomaly scores are computed via log-likelihood.

In parallel, prediction-based methods have evolved to leverage both sequential modeling and skeletal structure. Prior to the widespread adoption of GCNs, Fan et al. [160] used a combination of feedforward and recurrent (GRU) networks for future pose prediction. More recent works incorporate GCNs to simultaneously capture spatial (joint connectivity) and temporal (movement trajectory) patterns [161]. To further enrich the input representation, some researchers propose decomposing the pose into local (individual motion) and global (interpersonal interaction) components, as seen in [158], [159], leading to a better understanding of both individual and group behavior. Alternatively, Rodrigues et al. [157] introduce a multi-timescale approach, predicting both past and future frames at varying temporal resolutions to effectively capture both short-term and long-term anomalies.

To boost overall performance, several studies adopt multibranch architectures that combine reconstruction and prediction tasks. These systems benefit from complementary perspectives: reconstruction captures spatial structure while prediction leverages temporal dynamics. For instance, GRUbased [167], LSTM-based [166], and transformer-based [168], [170] multi-branch models all report improved performance by sharing an encoder while diverging into task-specific decoders. Li et al. [165] enhance this design by incorporating adversarial training, aligning with trends in pixel-based VAD to improve the quality of generated sequences. Additionally, Noghre et al. [42] propose a hybrid model that combines variational autoencoding for distribution-based scoring with a trajectory prediction branch, demonstrating the advantage of unifying multiple learning objectives under a coherent architecture.

Overall, pose-based VAD methods are evolving toward architectures that jointly model structure, motion, and probability, offering a privacy-aware and semantically rich alternative to pixel-level approaches. The integration of reconstruction, prediction, and distribution modeling, along with architectural

TABLE V OVERVIEW OF POSE -BASED APPROACHES IN VAD FOR PUBLIC SAFETY. * DENOTES THAT MULTIPLE ALTERNATIVE ARCHITECTURES HAVE BEEN USED .

| Approach           | Supervision          | Strategy    | Architecture     | Distinct Characteristics / Novel Contributions                                                                                                                                                                                                 |
|--------------------|----------------------|-------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MoPRL [70]         | Self/Semi-supervised | S3          | Transforme       | Uses a motion embedder followed by a spatio-temporal transforme
 for reconstruction, leveraging motion priors extracted through
 first-order difference statistics
 Uses 1D convolutions to predict past and future poses at multiple          |
| [157]              | Self/Semi-supervised | S4          | CNN              | Uses 1D convolutions to predict past and future poses at multiple
 timescales, capturing short- and long-term anomalies without relyin
 on fixed observation windows
 Combines hierarchical spatiotemporal graphs with a twobranch             |
| STGformer [158]    | Self/Semi-supervised | S4          | GCN,
 Transforme | Combines hierarchical spatio-temporal graphs with a two-branch
 architecture (local and global prediction) using spatial and tempor
 Transformers alongside GCNs                                                                               |
| HSTGCNN [159]      | Self/Semi-supervised | S4          | GCN, CNN         | Uses hierarchical spatio-temporal graphs with local and global
 prediction branches, applying 2D temporal followed by 2D spatia
 graph convolutions
 CNN il fhilGRU l                                                                          |
| [160]              | Self/Semi-supervised | S4          | CNN, GRU         | NN extracts spatial features, while GRU captures temporal
 dependencies
 il h lifdiid di                                                                                                                                                       |
| Normal Graph [161] | Self/Semi-supervised | S4          | GCN              | es spatiotemporal graph convolution for prediction and deriv
 anomaly scores from the prediction loss
 ltliiltt Giitd iti                                                                                                                      |
| PoseCVAE [162]     | Self/Semi-supervised | S4          | CNN              | imulates anomalies via latent Gaussian mixtures, and is trained
 through a three-stage process combining reconstruction,
 KL-divergence, and binary cross-entropy losses                                                                       |
| [163]              | Self/Semi-supervised | S6          | GCN              | An autoencoder is used for feature extraction, with latent space
 clustering for final detection
 liiflill diibi                                                                                                                               |
| STG-NF [69]        | Self/Semi-supervised | S6          | GCN              | Uses normalizing flow to map inputs to a latent normal distributio
 computing normality scores via likelihood and minimizing negativ
 log-likelihood during training                                                                           |
| GEPC [164]         | Self/Semi-supervised | S6          | GCN              | encoder is used for feature extraction, with latent sp
 clustering applied for VAD
 titl h ltiiVAE tt                                                                                                                                          |
| TSGAD [42]         | Self/Semi-supervised | S6          | GCN              | Leverages spatio-temporal graph convolution in a VAE structure
 using the distance from the latent mean and variance to score
 anomalies based on deviation from the learned normal distributio                                                |
| MemWGAN-GP
 [165]  | Self/Semi-supervised | S3, S4      | CNN              | A single-encoder dual-decoder generator with a critic, reconstructin
 past and predicts future sequences via memory-augmented branche                                                                                                          |
| STGCAE-LSTM
 [166] | Self/Semi-supervised | S3, S4      | GCN,
 LSTM       | ppqyg
 Single-encoder, dual-decoder architecture with LSTM in the latent
 space for enhanced temporal analysis                                                                                                                                 |
| MPED-RNN [167]     | Self/Semi-supervised | S3, S4      | GRU              | es global-local decomposition with a single encoder and dual
 GRU-based decoders
 ilddlddfdid                                                                                                                                                  |
| SPARTA [168]       | Self/Semi-supervised | S3, S4      | Transformer      | Features a single-encoder, dual-decoder transformer design and
 introduces a novel pose tokenization method by incorporating
 relative movement to emphasize motion dynamics
 patio-temporal GCN and attention are used for reconstruction, wi |
| MSTA-GCN [169]     | Self/Semi-supervised | S3, S6      | GCN              | Spatio-temporal GCN and attention are used for reconstruction, wit
 both reconstruction and latent space clustering                                                                                                                            |

TABLE VI OVERVIEW OF VEHICLE -CENTRIC VAD APPROACHES .

| Task                | Approach          | Supervision          | Strategy    | Architectur             | Distinct Characteristics / Novel Contributions                                                                                                                                                                                                                                                     |
|---------------------|-------------------|----------------------|-------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Task 
 Surveillance | [171]             | Supervised           | S1          | -                       | The system detects five types of traffic anomalies (speeding,
 one-way violations, overtaking, illegal parking, and improper
 drop-offs) by combining deep learning for object detection and
 tracking with handcrafted algorithms
 YOLOv5 is used for object detectionwhile anomalies are detecte |
| Task 
 Surveillance | [172]             | Supervised           | S1          | Decision Tree           | YOLOv5 is used for object detection, while anomalies are detected
 using decision trees                                                                                                                                                                                                            |
| Task 
 Surveillance | [21]              | Supervised           | S1          | CNN                     | s frames as accident or non-accident using a rolling a
 prediction algorithm                                                                                                                                                                                                                       |
|                     | DiffTAD
 [173]    | Self/Semi-supervised | S3          | Transformer             | Models anomalies as a noisytonormal reconstruction process usin
 Denoising Diffusion Probabilistic Models (DDPM), integrating
 Transformer-based temporal and spatial encoders to capture
 inter-vehicle dynamics                                                                                  |
|                     | VegaEdge
 [174]   | Self/Semi-supervised | S4          | GIN                     | A pipeline from object detection to trajectory prediction detect
 anomalies by comparing expected and actual trajectories
 rajectories are encoded into smoothed feature vectors with first a                                                                                                      |
|                     | [175]             | Self/Semi-supervised | S6          | SOM                     | Trajectories are encoded into smoothed feature vectors with first a
 second-order motion information, allowing the SOM to detect
 unusual behavior by learning the distribution of normal trajectorie                                                                                              |
|                     | [176]             | Unsupervised         | S7          | -                       | clustering method based on K-means, modeling each motion pattern
 as a chain of Gaussian distributions, and enabling both anomaly
 detection and behavior prediction
 A hihil Bifk hLDA d HDP                                                                                                      |
|                     | [177]             | Unsupervised         | S8          | Bayesian Model          | A hierarchical Bayesian framework that uses LDA and HDP to
 jointly model atomic activities and multi-agent interactions withou
 requiring labeled data
 Converts vehicle trajectories into gradient imagesleverages a CNN                                                                         |
|                     | [178]             | Self/Semi-supervised | S3, S6      | CNN                     | Converts vehicle trajectories into gradient images, leverages a CNN
 to classify normal trajectories via unsupervised clustering, and uses 
 VAE to detect unseen anomalies through reconstruction loss                                                                                            |
| Autonomous Driving  | [179]             | Supervised           | S1          | CNN, MLP
 SVM           | Temporal features are clustered after MLP processing to identif
 potential accidents, which are then combined with CNN-based
 spatial features and classified using an SVM                                                                                                                         |
| Autonomous Driving  | [180]             | Supervised           | S1          | CNN                     | Combines YOLOv5, lane alignment, and motion tracking to det
 td hil                                                                                                                                                                                                                                |
| Autonomous Driving  | TempoLearn
 [181] | Supervised           | S1          | CNN, LSTM,
 Transformer | pp
 ses CNN and LSTM for spatiotemporal feature extraction and a
 Transformer classifier for accident detection                                                                                                                                                                                    |
| Autonomous Driving  | [182]             | Self/Semi-supervised | S4          | CNN, LSTM               | ollaborative multi-task framework for jointly predicting fut
 frames, object locations, and scene context
 fd jid hl j                                                                                                                                                                             |
| Autonomous Driving  | FOL [183]         | Self/Semi-supervised | S4          | CNN                     | e expected trajectory is compared to the actu
 for detecting abnormal behaviors                                                                                                                                                                                                                    |
| Autonomous Driving  | [184]             | Self/Semi-supervised | S4          | Transformer             | A dual GAN framework with a Swin-Unet-based generator to predi
 intermediate frames using both optical flow and cropped inputs
 Combines memoryaugmented autoencoders for reconstruction an                                                                                                        |
| Autonomous Driving  | HF2-VAD
 [185]    | Self/Semi-supervised | S3, S4      | CNN                     | Combines memoryaugmented autoencoders for reconstruction and
 onditional VAEs for future frame prediction, enabling fine-grained
 dense anomaly localization                                                                                                                                       |
| Autonomous Driving  | [186]             | Self/Semi-supervised | S3, S6      | 3D CNN, GCN             | Proposes two models: one based on manifold learning to identify
 out-of-distribution anomalies, and another using reconstruction to
 detect deviations from normal data                                                                                                                            |

innovations such as GCNs and transformer positions posebased methods as a robust and scalable direction in VAD.

## VI. VEHICLE-CENTRIC VAD

## A. Road Surveillance

In the context of vehicle VAD for road surveillance, systems are primarily utilized by traffic monitoring authorities and urban infrastructure. These systems demand high-resolution spatial coverage, real-time or near-real-time processing, and robustness under diverse environmental conditions. The corresponding responses are typically passive and retrospective, including alert generation, traffic violation reporting, or data archiving for forensic purposes.

Early methods in vehicle-focused VAD followed similar trajectories to general VAD research, relying on handcrafted features and rule-based logic (see Table VI). These approaches often combine object detection and tracking with non-deep learning models for classification. Pramanik et al. [171] employed five distinct algorithms to identify specific behaviors such as speed violations and illegal parking. Zhou et al. [179]

utilized Support Vector Machines (SVMs) for accident detection based on extracted spatial-temporal features. Although effective for narrowly defined tasks, these approaches lack flexibility and generalization to unforeseen behaviors. While early deep learning-based models such as [21] extended detection capabilities, they still largely operated within constrained anomaly categories.

To mitigate these limitations, Aboah et al. [172] proposed a decision-tree-based method that evaluates foreground and background object detections using spatial thresholds and Intersection-over-Union (IoU) metrics, offering a more adaptable and interpretable rule-based framework.

A more flexible perspective involves casting anomaly detection as an unsupervised clustering problem, where anomalies are treated as statistical outliers. These methods utilize features such as motion trajectories, foreground activity, or background dynamics to learn normal patterns without requiring explicit labels. Hu et al. [176] applied k-means clustering to vehicle trajectories, while Niebles et al. [177] adopted hierarchical Bayesian models—originally developed for language modeling—to cluster interactions and motions, thereby learning the

TABLE VII OVERVIEW OF REVIEWED WORK IN FIRE AND FLOOD DETECTION. ARCHITECTURES ARE INFERRED FROM REPORTED METHODOLOGIES WHEN POSSIBLE .

| Task            | Approach        | Architecture         | Distinct Characteristics / Novel Contribution                                                                                                                                                                                                                                                                                                         |
|-----------------|-----------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                 | [187]           | GMM                  | Uses GMM motion detection and a region growth tracking, enabling accurate fire
 segmentation, growth rate estimation, and significant false alarm reduction
 Integrates 2D Haar wavelet transforms with convolutional neural networks to combi
 spatial and spectral features, achieving higher detection accuracy and significantly
 diflld ttil lit |
|                 | [22]            | CNN                  | segmentation, growth rate estimation, and significant false alarm reduction
 Integrates 2D Haar wavelet transforms with convolutional neural networks to combin
 spatial and spectral features, achieving higher detection accuracy and significantly
 reducing false alarms and computational complexity
 A fk biiidiih ilid id                      |
|                 | Fire-Det [24]   | CNN                  | A two-stage framework combining motion detection with specialized Fire-Det an
 lightweight Fire-Det Nano models, enabling fast, accurate early fire detection
 ased on EfficientNetB0 enhanced with stacked autoencoders and dense connectio                                                                                                          |
|                 | [189]           | CNN                  | achieving high accuracy, reduced false alarms, and efficient real-time inferencing
 Combining EfficientNet and YOLOv5, leveraging compound scaling and real-tim
 bjdi                                                                                                                                                                                 |
|                 |                 |                      | object detection
 modified YOLOv5 model within an edge computing framework using Jetson Nan
 featuring a dropout-enhanced architecture for improved accuracy and speedand                                                                                                                                                                             |
|                 | [191]           | CNN                  | featuring a dropoutenhanced architecture for improved accuracy and speed, and
 integration with cloud services for real-time alerting
 n improved YOLOv5s-based fire detection model that enhances detection accuracy                                                                                                                                 |
|                 | [191]           | CNN                  | and efficiency by integrating CBAM, BiFPN, and transposed convolution
 LiWihFi(LWF) fk blfl                                                                                                                                                                                                                                                           |
|                 | [192]           | CNN                  | y y gg , , p
 oposes Learning Without Forgetting (LWF) framework to enable transfer lea                                                                                                                                                                                                                                                               |
|                 | [193]           | CNN                  | pg gg 
 nalyzes the efficacy of famous networks such as AlexNet, GoogLeNet, and VGG                                                                                                                                                                                                                                                                   |
|                 | [195]           | CNN                  | p g
 models YOLOv8 and VQ-VAE, achieving high precision and robustness
 Combines transfer learning with YOLOv8 and the TranSDet model, incorporating                                                                                                                                                                                                  |
|                 | [196]           | CNN                  | Integrates a dehazing algorithm with a fine-tuned YOLO-v10 for ship fire detectio                                                                                                                                                                                                                                                                     |
|                 | [197]           | 3D CNN               | A modified MobileNetV3 integrated with a 3D CNN and a novel soft attention
 mechanism, enhancing spatial awareness and reducing model complexity
 A ViiTfbd dl iilfid if                                                                                                                                                                              |
|                 | FWSRNet [198]   | Transformer          | mechanism, enhancing spatial awareness and reducing model complexity
 A Vision Transformer-based model incorporating self-attention and contrastive feat
 learning for fine-grained wildfire and smoke recognition                                                                                                                                    |
|                 | [199]           | Transformer          | learning for fine-grained wildfire and smoke recognition
 A modified vision transformer architecture for fire detection that enables learnin
 from scratch on small to medium-sized datasets by integrating shifted patch
 tokenisation and locality selfattention                                                                                    |
| Flood Detection | [26]            | -                    | Combines background subtraction, morphological operations, color probability
 modeling across, and spatial features like edge density and boundary roughnes
 robabilistic model for flood detection by combining spatial features with temp                                                                                                           |
| Flood Detection | [200]           | Bayesian
 Classifier | modeling across, and spatial features like edge density and boundary roughness
 A probabilistic model for flood detection by combining spatial features with temp
 variation and a non-central chi-square-based positional prior, using Bayes                                                                                                         |
| Flood Detection | V-FloodNet [25] | CNN                  | variation and a noncentral chisquarebased positional prior, usin
 classification and patch-level scoring
 A video segmentation system that uses template-matching-based w                                                                                                                                                                             |
| Flood Detection | FRAD [201]      | CNN                  | Applies a CNN network to high-resolution multispectral remote sensing ima
 (SPOT-5) for supervised classification of urban flood risk
 YOLO4bd dlihd fbfld dh iii                                                                                                                                                                                     |
| Flood Detection | [202]           | CNN                  | Applies a CNN network to high-resolution multispectral remote sensing image
 (SPOT-5) for supervised classification of urban flood risk
 A YOLOv4-based deep learning method for urban flood depth estimation using tr                                                                                                                                |
| Flood Detection |                 | CNN                  | p g pg 
 images, leveraging submerged reference objects
 Flood severity classification from videos, combining spatial features extracted by                                                                                                                                                                                                           |
| Flood Detection | [203]           | CNN, GRU             | Flood severity classification from videos, combining spatial features extracted by 
 modified VGG16-based CNN with temporal dependencies captured by GRUs
 Fld dttiiil dibbiiil ftifitd                                                                                                                                                               |
| Flood Detection | [204]           | CNN,
 LSTM           | Flood detection in social media by combining visual features using a fine-tune
 InceptionV3 CNN with semantic features from metadata using a bidirectional LS                                                                                                                                                                                         |

structure of normal behavior in a probabilistic manner. Such clustering-based methods offer greater adaptability in complex or evolving environments.

More recent work has shifted toward deep learning models that emphasize generalization. Many of these methods fall into the prediction/reconstruction-based paradigms discussed in Section III. Santhosh et al. [178] employed a variational autoencoder to reconstruct trajectory data, while Li et al. [173] leveraged diffusion models to learn the data distribution. In the prediction domain, Katariya et al. [174] used a graph isomorphism network with attention mechanisms to model interactions and forecast future trajectories, and Fang et al. [182] proposed a multi-task learning framework that predicts future frames, object locations, and scene context simultaneously.

## B. Autonomous Driving

In autonomous driving, Vehicle VAD serves as a realtime, safety-critical component integrated into the vehicle's decision-making pipeline. These systems demand low-latency processing, precise detection of complex motion patterns, and seamless integration with multi-sensor fusion modules including LiDAR, radar, and cameras.

Early research in this domain primarily focused on detecting well-defined types of anomalies, often formulated as supervised classification (see Table VI). Park et al. [180] addressed the detection of stopped vehicles using dense optical flow to estimate host vehicle motion and bounding-box analysis to track surrounding vehicles. Similarly, some works have narrowed their scope to the detection and categorization of different types of collisions. Htun et al. [181] proposed a deep learning architecture that uses CNNs and LSTMs to extract spatial and temporal features, respectively, followed by a region proposal module and a classification head to detect and categorize collision types.

Building upon these constrained approaches, more flexible systems have emerged. Zhou et al. [179] introduced a two-stage coarse-to-fine framework: the first stage performs clustering of encoded temporal features to identify outlier frames as potential anomaly candidates, while the second stage applies object-level spatial feature extraction and a trained

Y Axis

Fig. 7. Severity of Each Challenge Across Different VAD Domains.

SVM classifier to confirm accident frames.

Reconstruction-based methods have also gained traction due to their generalization capacity to unseen anomalies. Haresh et al. [186] enhanced traditional autoencoder architectures by incorporating region proposal networks for object detection and graph convolutional networks (GCNs) to model object interactions, improving the semantic richness of reconstructions.

One of the most adopted modalities is optical flow, which provides dense motion information. Optical flow enables the detection of sudden or abnormal motion patterns, making it useful across both prediction and reconstruction-based paradigms. Bogdoll et al. [185] proposed a convolutional variational autoencoder that fuses features from both RGB and optical flow, improving anomaly reconstruction. In predictionbased frameworks, Yao et al. [183] leveraged optical flow for future object localization and ego-motion prediction, detecting anomalies based on deviations from expected motion trajectories. Ru et al. [184] extended this idea through a dualGAN framework that jointly predicts both optical flow and appearance features in regions of interest using a Swin-Unet backbone, achieving high accuracy at the cost of computational efficiency. Similarly, Fang et al. [182] proposed a multi-task framework incorporating future frame prediction, motion trajectory consistency, and visual context modeling, with optical flow as a core feature to enhance anomaly detection performance.

Despite the demonstrated success of prediction-based models, their performance can degrade in highly dynamic environments involving complex multi-agent interactions or unexpected environmental changes. These limitations are particularly pronounced in ego-centric settings, where the camera is mounted on a moving vehicle, increasing the risk of false positives due to background motion or occlusions.

## VII. ENVIRONMENTAL-CENTERIC VAD

Environmental VAD is critical for enabling rapid response and minimizing harm, particularly during the real-time detection stage of disaster management. Unlike prediction and postevent assessment, which rely on early indicators or support recovery and fall outside the scope of this work, real-time detection has been primarily approached through supervised methods that treat specific disasters as classification tasks. Video-based fire and flood detection have received the most attention in computer vision due to their structured visual signatures, whereas disasters like hurricanes and tsunamis are better suited to satellite imagery, and events like earthquakes

![Image](artifacts/image_000006_3e498f348643bf533e2d2a7d5b7a1008ae42bef415b3a1cdbae73e048537c4f9.png)

X Axis and droughts often lack distinct visual cues. This survey focuses on video-based methods for detecting floods and fires (see Table VII), where vision remains a central and effective modality.

## A. Fire Detection

CNN-based methods have been explored for fire detection due to their ability to capture spatial features. Early works utilized pretrained CNN models such as MobileNetV2 [205], AlexNet [206], and GoogLeNet [207], fine-tuning them for fire detection tasks. Some studies combined these models with additional techniques to enhance performance. For instance, wavelet transforms were used to extract critical spectral features [22], while transfer learning with "learning without forgetting" ensured models retained prior knowledge when adapting to new environments [193]. Advanced approaches integrated 3D CNNs with modified attention mechanisms to improve accuracy and employed Grad-CAM for visual interpretability of model decisions [197].

Another group of studies [189], [190], [195], [196] defined fire detection as a subset of object detection, leveraging and adapting well-known algorithms such as Faster R-CNN [208] and YOLO [106] for this purpose. For instance, [191] extended YOLOv5s by incorporating Convolutional Block Attention Modules (CBAM) [209] for improving feature fusion and replacing nearest neighbor interpolation with transposed convolution, introducing a fast, compact model and a more complex, accurate version tailored for fire detection. Similarly, [194] utilized YOLOv8 as a feature extractor to identify regions likely to contain fire. While this approach alone can serve as a fire detection method, they augmented it with a Vector Quantized Autoencoder (VQ-VAE) [210] to model the distribution of fire patterns, thereby providing an additional layer of analysis to reduce false positives and enhance detection reliability.

With the growing popularity of transformers, several studies have begun employing the Vision Transformers (ViT) [211] for fire detection. [199] introduces a specialized tokenization method designed for effective input tokenization for transformers. Additionally, [198] utilizes a contrastive feature learning mechanism to enhance the model's discriminative capabilities.

Earlier works treated video as independent frames, ignoring temporal dynamics. Recent studies, however, emphasize the importance of motion. For example, [187] uses segmentation and GMMs to identify flame-like motion and estimate fire growth, while [24] applies GMMs for motion filtering before

fire classification. These approaches demonstrate how incorporating temporal cues enhances accuracy and context awareness.

## B. Flood Detection

Earlier works [26], [200] rely on fundamental probabilistic models and heuristic approaches. These methods focus on extracting visual features like color, texture, and motion to identify flood regions. [200] integrates color, texture, and dynamic features within a probabilistic framework, leveraging spatial distributions to enhance detection accuracy. [26] employs background subtraction, morphological processing, and boundary roughness analysis for improved efficacy.

Deep learning brought strong advancements [201], [203], [204]. [203] utilizes a hybrid CNN-GRU model to classify flood severity in videos, combining spatial feature extraction with temporal modeling of sequential frames. [201] adopts a CNN-based Flood-Risk Assessment and Detection (FRAD) method for processing multispectral satellite images to identify flood-risk zones, emphasizing urban planning applications. [204] combines visual CNN-based analysis with a BiLSTM network for textual metadata processing, creating a multimodal approach to flood detection. These approaches exemplify the power of deep learning in capturing both spatial and temporal intricacies, demonstrating significant improvements over traditional models in accuracy and versatility.

More advanced architectures [25], [202], [212], incorporate SOTA techniques to enhance flood detection capabilities. [25] proposes V-FloodNet, a system integrating video segmentation (AFB-URR) and image segmentation (EfficientNet-B4 and LinkNet) with novel template-matching for depth estimation. [212] introduces DX-FloodLine, combining VGG16-LSTM for flood classification and Faster R-CNN with Mask R-CNN for object detection. [202] applies YOLOv4 for urban flood detection, utilizing traffic images with submerged reference objects and achieving real-time performance.

## VIII. CONCLUSIONS AND FUTURE DIRECTIONS

This survey provides a comprehensive and structured overview of deep learning-based Video Anomaly Detection (VAD), examining major challenges, learning paradigms, and a range of application domains. By incorporating human, vehicle-, and environment-centric perspectives, it reveals both shared foundations and domain-specific characteristics, facilitating meaningful cross-domain insights. The proposed taxonomy of supervision levels and adaptive strategies clarifies the strengths and limitations of existing methods, offering actionable guidance for designing effective VAD systems. In identifying critical research gaps, this work outlines promising directions for future exploration and serves as both a primer for newcomers and a valuable reference for researchers seeking to build robust, scalable solutions for real-world applications. Building on the challenges and trends observed across different VAD domains, we further evaluate the severity of open problems, as visualized in Figure 7, to support strategic research planning. Environment-centric VAD tends to be more manageable due to its structured, constrained settings. In contrast, autonomous driving remains highly challenging due to issues like domain shift, real-time performance, and sensor calibration (C8, C14, C16). Large-scale deployment in road surveillance and public safety introduces major scalability concerns (C15), driving the development of resource-efficient models, along with alternative data modalities. In healthcare, annotation remains a significant bottleneck (C2) due to the dependence on expert knowledge, underscoring the importance of label-efficient approaches such as few-shot and weakly supervised learning. Moreover, data scarcity (C1) persists across nearly all domains, prompting increased interest in synthetic data generation, especially with generative AI to simulate anomalies and boost model robustness.

## ACKNOWLEDGMENTS

This research is funded by the United States National Science Foundation (NSF) under award number 2329816.

## REFERENCES

- [1] K. Shaukat et al., "A review of time-series anomaly detection techniques: A step to future perspectives," in Advances in information and communication: proceedings of the 2021 future of information and communication conference (FICC), volume 1. Springer, 2021, pp. 865–877.
- [2] J. Liu et al., "Deep industrial image anomaly detection: A survey," Machine Intelligence Research, vol. 21, pp. 104–135, 2024.
- [3] P. Mishra et al., "Vt-adl: A vision transformer network for image anomaly detection and localization," in 2021 IEEE 30th International Symposium on Industrial Electronics (ISIE). IEEE, 2021, pp. 01–06.
- [4] J. Yang et al., "Visual anomaly detection for images: A systematic survey," Procedia computer science, vol. 199, pp. 471–478, 2022.
- [5] A. D. Pazho et al., "A survey of graph-based deep learning for anomaly detection in distributed systems," IEEE Trans. Knowl. Data Eng. , vol. 36, pp. 1–20, 2023.
- [6] K. Rezaee et al., "A survey on deep learning-based real-time crowd anomaly detection for secure distributed video surveillance," Personal and Ubiquitous Computing, vol. 28, pp. 135–151, 2024.
- [7] D. Fahrmann ¨ ¨ et al., "Anomaly detection in smart environments: a comprehensive survey," IEEE access, 2024.
- [8] Y. A. Samaila et al., "Video anomaly detection: A systematic review of issues and prospects," Neurocomputing, p. 127726, 2024.
- [9] P. K. Mishra et al., "Skeletal video anomaly detection using deep learning: Survey, challenges, and future directions," IEEE Trans. Emerg. Topics Comput., 2024.
- [10] A. D. Pazho et al., "Ancilia: Scalable intelligent video surveillance for the artificial intelligence of things," IEEE Internet Things J., vol. 10, pp. 14 940–14 951, 2023.
- [11] X. Yang et al., "Deep learning technologies for time series anomaly detection in healthcare: A review," Ieee Access, vol. 11, pp. 117 788– 117 799, 2023.
- [12] A. A. Ali et al., "Anomaly detection in healthcare monitoring survey," in Advanced Research Trends in Sustainable Solutions, Data Analytics, and Security. IGI Global Scientific Publishing, 2025, pp. 29–56.
- [13] T. Fernando et al., "Deep learning for medical anomaly detection–a survey," ACM Computing Surveys (CSUR), vol. 54, pp. 1–37, 2021.
- [14] Y. M. Galvao˜ ˜ et al., "Anomaly detection in smart houses for healthcare: Recent advances, and future perspectives," SN Computer Science , vol. 5, p. 136, 2024.
- [15] D. Bogdoll et al., "Anomaly detection in autonomous driving: A survey," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4488–4499.
- [16] S. Baccari et al., "Anomaly detection in connected and autonomous vehicles: A survey, analysis, and research challenges," IEEE Access , vol. 12, pp. 19 250–19 276, 2024.
- [17] D. Bogdoll et al., "Perception datasets for anomaly detection in autonomous driving: A survey," in 2023 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2023, pp. 1–8.
- [18] J. R. V. Solaas et al., "Systematic literature review: Anomaly detection in connected and autonomous vehicles," IEEE Trans. Intell. Transp. Syst., 2024.

- [19] K. K. Santhosh et al., "Anomaly detection in road traffic using visual surveillance: A survey," Acm Computing Surveys (CSUR), vol. 53, pp. 1–26, 2020.
- [20] M. Rathee et al., "Automated road defect and anomaly detection for traffic safety: a systematic review," Sensors, vol. 23, p. 5656, 2023.
- [21] S. W. Khan et al., "Anomaly detection in traffic surveillance videos using deep learning," Sensors, vol. 22, p. 6563, 2022.
- [22] L. Huang et al., "Fire detection in video surveillances using convolutional neural networks and wavelet transform," Engineering Applications of Artificial Intelligence, vol. 110, p. 104737, 2022.
- [23] A. Saleh et al., "Forest fire surveillance systems: A review of deep learning methods," Heliyon, vol. 10, 2024.
- [24] S. Gao et al., "Two-stage deep learning-based video image recognition of early fires in heritage buildings," Engineering Applications of Artificial Intelligence, vol. 129, p. 107598, 2024.
- [25] Y. Liang et al., "V-floodnet: A video segmentation system for urban flood detection and quantification," Environmental Modelling &amp; Software , vol. 160, p. 105586, 2023.
- [26] A. Filonenko et al., "Real-time flood detection for video surveillance," in IECON 2015-41st annual conference of the IEEE industrial electronics society. IEEE, 2015, pp. 004 082–004 085.
- [27] L. Lopez-Fuentes et al., "Review on computer vision techniques in emergency situations," Multimedia Tools and Applications, vol. 77, pp. 17 069–17 107, 2018.
- [28] B. R. Ardabili et al., "Understanding policy and technical aspects of ai-enabled smart video surveillance to address public safety," Computational Urban Science, vol. 3, p. 21, 2023.
- [29] B. Rahimi Ardabili et al., "Understanding ethics, privacy, and regulations in smart video surveillance for public safety," arXiv preprint arXiv:2212.12936, 2022.
- [30] A. D. Pazho et al., "Vt-former: An exploratory study on vehicle trajectory prediction for highway surveillance through graph isomorphism and transformer," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5651–5662.
- [31] B. R. Ardabili et al., "Exploring public's perception of safety and video surveillance technology: A survey approach," Technology in Society , vol. 78, p. 102641, 2024.
- [32] Y. Tian et al., "Weakly-supervised video anomaly detection with robust temporal feature magnitude learning," in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 4975–4986.
- [33] P. Wu et al., "Vadclip: Adapting vision-language models for weakly supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 6, 2024, pp. 6074– 6082.
- [34] X. Wang et al., "Robust unsupervised video anomaly detection by multipath frame prediction," IEEE Trans. Neural Netw. Learn. Syst. , vol. 33, pp. 2301–2312, 2021.
- [35] B. Ramachandra et al., "A survey of single-scene video anomaly detection," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, pp. 2293– 2312, 2020.
- [36] Y. Yang et al., "Follow the rules: reasoning for video anomaly detection with large language models," in European Conference on Computer Vision. Springer, 2024, pp. 304–322.
- [37] W. Liu et al., "Future frame prediction for anomaly detection – a new baseline," in 2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [38] R. Rodrigues et al., "Multi-timescale trajectory prediction for abnormal human activity detection," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), March 2020.
- [39] A. Danesh Pazho et al., "Chad: Charlotte anomaly dataset," in Scandinavian Conference on Image Analysis. Springer, 2023, pp. 50–66.
- [40] Y. Zhu et al., "Towards open set video anomaly detection," in European Conference on Computer Vision. Springer, 2022, pp. 395–412.
- [41] G. Alinezhad Noghre et al., "Understanding the challenges and opportunities of pose-based anomaly detection," in Proceedings of the 8th International Workshop on Sensor-Based Activity Recognition and Artificial Intelligence, 2023, pp. 1–9.
- [42] G. A. Noghre et al., "An exploratory study on human-centric video anomaly detection through variational autoencoders and trajectory prediction," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 995–1004.
- [43] S. Yao et al., "Evaluating the effectiveness of video anomaly detection in the wild: Online learning and inference for real-world deployment," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4832–4841.
- [44] Y. Zhu et al., "Context-aware activity recognition and anomaly detection in video," IEEE J. Sel. Topics Signal Process., vol. 7, pp. 91–101, 2012.
- [45] Y. Zhou et al., "Detecting anomaly in videos from trajectory similarity analysis," in 2007 IEEE international conference on multimedia and expo. IEEE, 2007, pp. 1087–1090.
- [46] B.-H. Wang et al., "Fall detection based on dual-channel feature integration," IEEE Access, vol. 8, pp. 103 443–103 453, 2020.
- [47] L. Gong et al., "A novel computer vision based gait analysis technique for normal and parkinson's gaits classification," in 2020 IEEE Intl Conf on Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech). IEEE, 2020, pp. 209–215.
- [48] B. Jin et al., "Diagnosing parkinson disease through facial expression recognition: video analysis," Journal of medical Internet research , vol. 22, p. e18697, 2020.
- [49] D. Mehta et al., "Privacy-preserving early detection of epileptic seizures in videos," in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 210–219.
- [50] L. Kirichenko et al., "Detection of shoplifting on video using a hybrid network," Computation, vol. 10, p. 199, 2022.
- [51] B. C. Das et al., "Efficient gun detection in real-world videos: Challenges and solutions," 2025.
- [52] K. Doshi et al., "Continual learning for anomaly detection in surveillance videos," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2020, pp. 254–255.
- [53] X. Huang et al., "Multi-level memory-augmented appearance-motion correspondence framework for video anomaly detection," in 2023 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2023, pp. 2717–2722.
- [54] M. Z. Zaheer et al., "Generative cooperative learning for unsupervised video anomaly detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 14 744–14 754.
- [55] Q. Li et al., "Essl: Enhanced spatio-temporal self-selective learning framework for unsupervised video anomaly detection." in ECAI, 2023, pp. 1398–1405.
- [56] K. Doshi et al., "Rethinking video anomaly detection-a continual learning approach," in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 3961–3970.
- [57] K. Faber et al., "Lifelong continual learning for anomaly detection: New challenges, perspectives, and insights," IEEE Access, vol. 12, pp. 41 364–41 380, 2024.
- [58] R. Jiao et al., "Survey on video anomaly detection in dynamic scenes with moving cameras," Artificial Intelligence Review, vol. 56, pp. 3515– 3570, 2023.
- [59] Z. Zamanzadeh Darban et al., "Deep learning for time series anomaly detection: A survey," ACM Computing Surveys, vol. 57, pp. 1–42, 2024.
- [60] Y. Lin et al., "A survey on rgb, 3d, and multimodal approaches for unsupervised industrial image anomaly detection," Information Fusion , p. 103139, 2025.
- [61] S. Olugbade et al., "A review of artificial intelligence and machine learning for incident detectors in road transport systems," Mathematical and Computational Applications, vol. 27, p. 77, 2022.
- [62] S. A. Ahmed et al., "Trajectory-based surveillance analysis: A survey," IEEE Trans. Circuits Syst. Video Technol., vol. 29, pp. 1985–1997, 2018.
- [63] A. Al-Lahham et al., "A coarse-to-fine pseudo-labeling (c2fpl) framework for unsupervised video anomaly detection," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , 2024, pp. 6793–6802.
- [64] M.-I. Georgescu et al., "Anomaly detection in video via self-supervised and multi-task learning," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12 742–12 752.
- [65] Z. Yang et al., "Context-aware video anomaly detection in long-term datasets," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 4002–4011.
- [66] P. Narwade et al., "Synthetic video generation for weakly supervised cross-domain video anomaly detection," in International Conference on Pattern Recognition. Springer, 2025, pp. 375–391.
- [67] A. Ponraj et al., "A video surveillance: Crowd anomaly detection and management alert system," Quantum Computing Models for Cybersecurity and Wireless Communications, pp. 139–152, 2025.
- [68] L. Luo et al., "Detecting and quantifying crowd-level abnormal behaviors in crowd events," IEEE Trans. Inf. Forensics Security, 2024.

- [69] O. Hirschorn et al., "Normalizing flows for human pose anomaly detection," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 13 545–13 554.
- [70] S. Yu et al., "Regularity learning via explicit distribution modeling for skeletal video anomaly detection," IEEE Trans. Circuits Syst. Video Technol., 2023.
- [71] K. Biradar et al., "Robust anomaly detection through transformerencoded feature diversity learning," in Proceedings of the Asian Conference on Computer Vision, 2024, pp. 115–128.
- [72] D. Bhardwaj et al., "Leveraging dual encoders with feature disentanglement for anomaly detection in thermal videos," in International Conference on Pattern Recognition. Springer, 2025, pp. 237–253.
- [73] D. Guo et al., "Ada-vad: Domain adaptable video anomaly detection," in Proceedings of the 2024 SIAM International Conference on Data Mining (SDM). SIAM, 2024, pp. 634–642.
- [74] Z. Wang et al., "Domain generalization for video anomaly detection considering diverse anomaly types," Signal, Image and Video Processing, vol. 18, pp. 3691–3704, 2024.
- [75] M. Cho et al., "Towards multi-domain learning for generalizable video anomaly detection," Advances in Neural Information Processing Systems, vol. 37, pp. 50 256–50 284, 2024.
- [76] R. Nawaratne et al., "Spatiotemporal anomaly detection using deep learning for real-time video surveillance," IEEE Transactions on Industrial Informatics, vol. 16, pp. 393–402, 2019.
- [77] M. M. Ali, "Real-time video anomaly detection for smart surveillance," IET Image Processing, vol. 17, pp. 1375–1388, 2023.
- [78] H. Karim et al., "Real-time weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2024, pp. 6848–6856.
- [79] S. Zhu et al., "Video anomaly detection for smart surveillance," in Computer Vision: A Reference Guide. Springer, 2021, pp. 1315–1322.
- [80] K. Doshi et al., "Online anomaly detection in surveillance videos with asymptotic bound on false alarm rate," Pattern Recognition, vol. 114, p. 107865, 2021.
- [81] J. Micorek et al., "Mulde: Multiscale log-density estimation via denoising score matching for video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 18 868–18 877.
- [82] Y. Nie et al., "Interleaving one-class and weakly-supervised models with adaptive thresholding for unsupervised video anomaly detection," in European Conference on Computer Vision. Springer, 2024, pp. 449–467.
- [83] A. Ntelopoulos et al., "Callm: Cascading autoencoder and large language model for video anomaly detection," in 2024 IEEE Thirteenth International Conference on Image Processing Theory, Tools and Applications (IPTA). IEEE, 2024, pp. 1–6.
- [84] B. Asal et al., "Ensemble-based knowledge distillation for video anomaly detection," Applied Sciences, vol. 14, p. 1032, 2024.
- [85] Y. Cai et al., "Medianomaly: A comparative study of anomaly detection in medical images," Medical Image Analysis, p. 103500, 2025.
- [86] Z. Z. Darban et al., "Dacad: Domain adaptation contrastive learning for anomaly detection in multivariate time series," arXiv preprint arXiv:2404.11269, 2024.
- [87] S. Wang et al., "Effective end-to-end unsupervised outlier detection via inlier priority of discriminative network," Advances in neural information processing systems, vol. 32, 2019.
- [88] G. Yu et al., "Deep anomaly discovery from unlabeled videos via normality advantage and self-paced refinement," in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition , 2022, pp. 13 987–13 998.
- [89] J. Lu et al., "Learning under concept drift: A review," IEEE Trans. Knowl. Data Eng., vol. 31, pp. 2346–2363, 2018.
- [90] S. Saurav et al., "Online anomaly detection with concept drift adaptation using recurrent neural networks," in Proceedings of the acm india joint international conference on data science and management of data , 2018, pp. 78–87.
- [91] S. Wu et al., "Adversarial sparse transformer for time series forecasting," Advances in neural information processing systems, vol. 33, pp. 17 105–17 115, 2020.
- [92] X. Tang et al., "Deep anomaly detection with ensemble-based active learning," in 2020 IEEE International Conference on Big Data (Big Data). IEEE, 2020, pp. 1663–1670.
- [93] S. Thrun et al., "Learning to learn: Introduction and overview," in Learning to learn. Springer, 1998, pp. 3–17.
- [94] Y. Lu et al., "Few-shot scene-adaptive anomaly detection," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16. Springer, 2020, pp. 125–141.
- [95] C. Finn et al., "Model-agnostic meta-learning for fast adaptation of deep networks," in International conference on machine learning . PMLR, 2017, pp. 1126–1135.
- [96] E. Hazan et al., "Introduction to online convex optimization," Foundations and Trends® in Optimization, vol. 2, pp. 157–325, 2016.
- [97] S. C. Hoi et al., "Online learning: A comprehensive survey," Neurocomputing, vol. 459, pp. 249–289, 2021.
- [98] S. Han et al., "Log-based anomaly detection with robust feature extraction and online learning," IEEE Trans. Inf. Forensics Security , vol. 16, pp. 2300–2311, 2021.
- [99] Z. Chen et al., "An effective cost-sensitive sparse online learning framework for imbalanced streaming data classification and its application to online anomaly detection," Knowledge and Information Systems , vol. 65, pp. 59–87, 2023.
- [100] S. Yao et al., "Evaluating the effectiveness of video anomaly detection in the wild: Online learning and inference for real-world deployment," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2024, pp. 4832–4841.
- [101] R. M. French, "Catastrophic forgetting in connectionist networks," Trends in cognitive sciences, vol. 3, pp. 128–135, 1999.
- [102] M. McCloskey et al., "Catastrophic interference in connectionist networks: The sequential learning problem," in Psychology of learning and motivation. Elsevier, 1989, vol. 24, pp. 109–165.
- [103] D. Cohn et al., "Improving generalization with active learning," Machine learning, vol. 15, pp. 201–221, 1994.
- [104] R. M. Monarch, Human-in-the-Loop Machine Learning: Active learning and annotation for human-centered AI. Simon and Schuster, 2021.
- [105] B. Settles, “Active learning literature survey,” 2009.
- [106] J. Redmon et al., "You only look once: Unified, real-time object detection," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 779–788.
- [107] E. Ilg et al., "Flownet 2.0: Evolution of optical flow estimation with deep networks," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2462–2470.
- [108] C. C. Loy et al., "Stream-based active unusual event detection," in Asian Conference on Computer Vision. Springer, 2010, pp. 161–175.
- [109] C. Change Loy et al., "Stream-based joint exploration-exploitation active learning," in 2012 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2012, pp. 1560–1567.
- [110] Y. Chen et al., "Vision-based fall event detection in complex background using attention guided bi-directional lstm," IEEE Access, vol. 8, pp. 161 337–161 348, 2020.
- [111] X. Cai et al., "Vision-based fall detection with multi-task hourglass convolutional auto-encoder," IEEE Access, vol. 8, pp. 44 493–44 502, 2020.
- [112] S. Chhetri et al., "Deep learning for vision-based fall detection system: Enhanced optical dynamic flow," Computational Intelligence, vol. 37, pp. 578–595, 2021.
- [113] W. Chen et al., "Fall detection based on key points of human-skeleton using openpose," Symmetry, vol. 12, p. 744, 2020.
- [114] O. Keskes et al., "Vision-based fall detection using st-gcn," IEEE Access, vol. 9, pp. 28 224–28 236, 2021.
- [115] J. Zhang et al., "Human fall detection based on body posture spatiotemporal evolution," Sensors, vol. 20, p. 946, 2020.
- [116] C. Khraief et al., "Elderly fall detection based on multi-stream deep convolutional networks," Multimedia Tools and Applications, vol. 79, pp. 19 537–19 560, 2020.
- [117] L. F. Gomez et al., "Improving parkinson detection using dynamic features from evoked expressions in video," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 1562–1570.
- [118] L. Gomez-Gomez et al., "Exploring facial expressions and affective domains for parkinson detection. arxiv 2020," arXiv preprint arXiv:2012.06563, 2012.
- [119] R. Kaur et al., "A vision-based framework for predicting multiple sclerosis and parkinson's disease gait dysfunctions—a deep learning approach," IEEE J. Biomed. Health Inform., vol. 27, pp. 190–201, 2022.
- [120] T. Connie et al., "Pose-based gait analysis for diagnosis of parkinson's disease," Algorithms, vol. 15, p. 474, 2022.
- [121] M. H. Monje et al., "Remote evaluation of parkinson's disease using a conventional webcam and artificial intelligence," Frontiers in neurology, vol. 12, p. 742654, 2021.
- [122] J. Archila et al., "A multimodal parkinson quantification by fusing eye and gait motion patterns, using covariance descriptors, from non-invasive computer vision," Computer methods and programs in biomedicine, vol. 215, p. 106607, 2022.

- [123] K. Sun et al., "Spatial attentional bilinear 3d convolutional network for video-based autism spectrum disorder detection," in ICASSP 20202020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 3387–3391.
- [124] S. Chen et al., "Attention-based autism spectrum disorder screening with privileged modality," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1181–1190.
- [125] M. Jiang et al., "Learning visual attention to identify people with autism spectrum disorder," in Proceedings of the ieee international conference on computer vision, 2017, pp. 3267–3276.
- [126] Y. Tao et al., "Sp-asdnet: Cnn-lstm based asd classification model using observer scanpaths," in 2019 IEEE International conference on multimedia &amp; expo workshops (ICMEW). IEEE, 2019, pp. 641–646.
- [127] A. Ali et al., "Video-based behavior understanding of children for objective diagnosis of autism," in VISAPP 2022-17th International Conference on Computer Vision Theory and Applications, 2022.
- [128] C. Wu et al., "Machine learning based autism spectrum disorder detection from videos," in 2020 IEEE International Conference on Ehealth Networking, Application &amp; Services (HEALTHCOM). IEEE, 2021, pp. 1–6.
- [129] N. Kojovic et al., "Using 2d video-based pose estimation for automated prediction of autism spectrum disorders in young children," Scientific Reports, vol. 11, p. 15069, 2021.
- [130] Y. Yang et al., "Video-based detection of generalized tonic-clonic seizures using deep learning," IEEE J. Biomed. Health Inform., vol. 25, pp. 2997–3008, 2021.
- [131] J.-C. Hou et al., "A self-supervised pre-training framework for visionbased seizure classification," in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 2022, pp. 1151–1155.
- [132] P. K. Pothula et al., "A real-time seizure classification system using computer vision techniques," in 2022 IEEE International Systems Conference (SysCon). IEEE, 2022, pp. 1–6.
- [133] D. Ahmedt-Aristizabal et al., "Vision-based mouth motion analysis in epilepsy: A 3d perspective," in 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2019, pp. 1625–1629.
- [134] C.-H. Chou et al., "Convolutional neural network-based fast seizure detection from video electroencephalograms," Biomedical Signal Processing and Control, vol. 80, p. 104380, 2023.
- [135] V. M. Garc¸ao˜ ˜ et al., "A novel approach to automatic seizure detection using computer vision and independent component analysis," Epilepsia , vol. 64, pp. 2472–2483, 2023.
- [136] D. Ahmedt-Aristizabal et al., "Motion signatures for the analysis of seizure evolution in epilepsy," in 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2019, pp. 2099–2105.
- [137] J.-C. Hou et al., "A multi-stream approach for seizure classification with knowledge distillation," in 2021 17th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). IEEE, 2021, pp. 1–8.
- [138] J. C. Hou et al., "Automated video analysis of emotion and dystonia in epileptic seizures," Epilepsy Research, vol. 184, p. 106953, 2022.
- [139] K. Simonyan et al., "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
- [140] K. He et al., "Deep residual learning for image recognition," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.
- [141] G. A. Mart ´ ´ınez-Mascorro et al., "Criminal intention detection at early stages of shoplifting cases by using 3d convolutional neural networks," Computation, vol. 9, p. 24, 2021.
- [142] I. Muneer et al., "Shoplifting detection using hybrid neural network cnn-bilsmt and development of benchmark dataset," Applied Sciences , vol. 13, p. 8341, 2023.
- [143] V. Manikandan et al., "A neural network aided attuned scheme for gun detection in video surveillance images," Image and Vision Computing , vol. 120, p. 104406, 2022.
- [144] M. T. Bhatti et al., "Weapon detection in real-time cctv videos using deep learning," Ieee Access, vol. 9, pp. 34 366–34 382, 2021.
- [145] T. Nyajowi et al., "Cnn real-time detection of vandalism using a hybridlstm deep learning neural networks," in 2021 IEEE AFRICON. IEEE, 2021, pp. 1–6.
- [146] Y. Yang et al., "Enhanced adversarial learning based video anomaly detection with object confidence and position," in 2019 13th International Conference on Signal Processing and Communication Systems (ICSPCS). IEEE, 2019, pp. 1–5.
- [147] D. Chen et al., "Nm-gan: Noise-modulated generative adversarial network for video anomaly detection," Pattern Recognition, vol. 116, p. 107969, 2021.
- [148] A. Barbalau et al., "Ssmtl++: Revisiting self-supervised multi-task learning for video anomaly detection," Computer Vision and Image Understanding, vol. 229, p. 103656, 2023.
- [149] G. Wang et al., "Video anomaly detection by solving decoupled spatiotemporal jigsaw puzzles," in European Conference on Computer Vision . Springer, 2022, pp. 494–511.
- [150] M. I. Georgescu et al., "A background-agnostic framework with adversarial training for abnormal event detection in video," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, pp. 4505–4523, 2021.
- [151] W. Luo et al., "Future frame prediction network for video anomaly detection," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, pp. 7505– 7520, 2021.
- [152] H. Shi et al., "Abnormal ratios guided multi-phase self-training for weakly-supervised video anomaly detection," IEEE Trans. Multimedia , vol. 26, pp. 5575–5587, 2023.
- [153] Q. Li et al., "Attention-based anomaly detection in multi-view surveillance videos," Knowledge-Based Systems, vol. 252, p. 109348, 2022.
- [154] C. Zhang et al., "Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16 271–16 280.
- [155] Z. Yang et al., "Text prompt with normality guidance for weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 899–18 908.
- [156] J. Salido et al., "Automatic handgun detection with deep learning in video surveillance images," Applied Sciences, vol. 11, p. 6085, 2021.
- [157] R. Rodrigues et al., "Multi-timescale trajectory prediction for abnormal human activity detection," in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2020, pp. 2626–2634.
- [158] C. Huang et al., "Hierarchical graph embedded pose regularity learning via spatio-temporal transformer for abnormal behavior detection," in Proceedings of the 30th ACM international conference on multimedia , 2022, pp. 307–315.
- [159] X. Zeng et al., "A hierarchical spatio-temporal graph convolutional neural network for anomaly detection in videos," IEEE Trans. Circuits Syst. Video Technol., vol. 33, pp. 200–212, 2021.
- [160] B. Fan et al., "Anomaly detection based on pose estimation and gruffn," in 2021 IEEE Sustainable Power and Energy Conference (iSPEC) . IEEE, 2021, pp. 3821–3825.
- [161] W. Luo et al., "Normal graph: Spatial temporal graph convolutional networks based prediction network for skeleton based video anomaly detection," Neurocomputing, vol. 444, pp. 332–337, 2021.
- [162] Y. Jain et al., "Posecvae: Anomalous human activity detection," in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021, pp. 2927–2934.
- [163] C. Liu et al., "A self-attention augmented graph convolutional clustering networks for skeleton-based video anomaly behavior detection," Applied Sciences, vol. 12, p. 4, 2021.
- [164] A. Markovitz et al., "Graph embedded pose clustering for anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 10 539–10 547.
- [165] n. Li et al., "Human-related anomalous event detection via memoryaugmented wasserstein generative adversarial network with gradient penalty," Pattern Recognition, vol. 138, p. 109398, 2023.
- [166] N. Li et al., "Human-related anomalous event detection via spatialtemporal graph convolutional autoencoder with embedded long shortterm memory network," Neurocomputing, vol. 490, pp. 482–494, 2022.
- [167] R. Morais et al., "Learning regularity in skeleton trajectories for anomaly detection in videos," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 11 996– 12 004.
- [168] G. A. Noghre et al., "Human-centric video anomaly detection through spatio-temporal pose tokenization and transformer," 2025.
- [169] X. Chen et al., "Multiscale spatial temporal attention graph convolution network for skeleton-based anomaly behavior detection," Journal of visual communication and image representation, vol. 90, p. 103707, 2023.
- [170] G. A. Noghre et al., "Posewatch: A transformer-based architecture for human-centric video anomaly detection using spatio-temporal pose tokenization," arXiv preprint arXiv:2408.15185, 2024.
- [171] A. Pramanik et al., "A real-time video surveillance system for traffic pre-events detection," Accident Analysis &amp; Prevention, vol. 154, p. 106019, 2021.

- [172] A. Aboah, "A vision-based system for traffic anomaly detection using deep learning and decision trees," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4207–4212.
- [173] C. Li et al., "Difftad: Denoising diffusion probabilistic models for vehicle trajectory anomaly detection," Knowledge-Based Systems, vol. 286, p. 111387, 2024.
- [174] V. Katariya et al., "Vegaedge: Edge ai confluence for real-time iotapplications in highway safety," Internet of Things, vol. 27, p. 101268, 2024.
- [175] J. Owens et al., "Application of the self-organising map to trajectory classification," in Proceedings Third IEEE International Workshop on Visual Surveillance. IEEE, 2000, pp. 77–83.
- [176] W. Hu et al., "A system for learning statistical motion patterns," IEEE Trans. Pattern Anal. Mach. Intell., vol. 28, pp. 1450–1464, 2006.
- [177] X. Wang et al., "Unsupervised activity perception by hierarchical bayesian models," in 2007 IEEE Conference on Computer Vision and Pattern Recognition, 2007, pp. 1–8.
- [178] K. K. Santhosh et al., "Vehicular trajectory classification and traffic anomaly detection in videos using a hybrid cnn-vae architecture," IEEE Trans. Intell. Transp. Syst., vol. 23, pp. 11 891–11 902, 2021.
- [179] Z. Zhou et al., "Spatio-temporal feature encoding for traffic accident detection in vanet environment," IEEE Transactions on Intelligent Transportation Systems, vol. 23, pp. 19 772–19 781, 2022.
- [180] J. Park et al., "Deep learning-based stopped vehicle detection method utilizing in-vehicle dashcams," Electronics, vol. 13, p. 4097, 2024.
- [181] S. S. Htun et al., "Tempolearn network: Leveraging spatio-temporal learning for traffic accident detection," IEEE Access, vol. 11, pp. 142 292–142 303, 2023.
- [182] J. Fang et al., "Traffic accident detection via self-supervised consistency learning in driving scenarios," IEEE Trans. Intell. Transp. Syst. , vol. 23, pp. 9601–9614, 2022.
- [183] Y. Yao et al., "Unsupervised traffic accident detection in first-person videos," in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 273–280.
- [184] H. Ru et al., "Enhanced anomaly detection in dashcam videos: Dual gan approach with swin-unet for optical flow and region of interest analysis," in 2024 International Joint Conference on Neural Networks (IJCNN). IEEE, 2024, pp. 1–8.
- [185] D. Bogdoll et al., "Hybrid video anomaly detection for anomalous scenarios in autonomous driving," arXiv preprint arXiv:2406.06423 , 2024.
- [186] S. Haresh et al., "Towards anomaly detection in dashcam videos," in 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020, pp. 1407–1414.
- [187] A. Khalil et al., "Fire detection using multi color space and background modeling," Fire technology, vol. 57, pp. 1221–1239, 2021.
- [188] H. Farman et al., "Efficient fire detection with e-efnet: A lightweight deep learning-based approach for edge devices," Applied Sciences , vol. 13, p. 12941, 2023.
- [189] S. Chitram et al., "Enhancing fire and smoke detection using deep learning techniques," Engineering Proceedings, vol. 62, p. 7, 2024.
- [190] A. S. Mahdi et al., "An edge computing environment for early wildfire detection," Annals of Emerging Technologies in Computing (AETiC) , vol. 6, pp. 56–68, 2022.
- [191] Z. Dou et al., "An improved yolov5s fire detection model," Fire Technology, vol. 60, pp. 135–166, 2024.
- [192] V. E. Sathishkumar et al., "Forest fire and smoke detection using deep learning-based learning without forgetting," Fire ecology, vol. 19, p. 9, 2023.
- [193] G. Son et al., "Video based smoke and flame detection using convolutional neural network," in 2018 14th International Conference on Signal-Image Technology &amp; Internet-Based Systems (SITIS). IEEE, 2018, pp. 365–368.
- [194] H. Zhao et al., "Fsdf: A high-performance fire detection framework," Expert Systems with Applications, vol. 238, p. 121665, 2024.
- [195] N. Yunusov et al., "Robust forest fire detection method for surveillance systems based on you only look once version 8 and transfer learning approaches," Processes, vol. 12, p. 1039, 2024.
- [196] F. Akhmedov et al., "Dehazing algorithm integration with yolo-v10 for ship fire detection," Fire, vol. 7, p. 332, 2024.
- [197] H. Yar et al., "An efficient deep learning architecture for effective fire detection in smart surveillance," Image and Vision Computing, vol. 145, p. 104989, 2024.
- [198] Y. Wang et al., "Computer vision-driven forest wildfire and smoke recognition via iot drone cameras," Wireless Networks, vol. 30, pp. 7603–7616, 2024.
- [199] H. Yar et al., "A modified vision transformer architecture with scratch learning capabilities for effective fire detection," Expert Systems with Applications, vol. 252, p. 123935, 2024.
- [200] P. V. K. Borges et al., "A probabilistic model for flood detection in video sequences," in 2008 15th IEEE International Conference on Image Processing. IEEE, 2008, pp. 13–16.
- [201] I. E. Villalon-Turrubiates, "Convolutional neural network for flood-risk assessment and detection within a metropolitan area," in 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS . IEEE, 2021, pp. 1339–1342.
- [202] P. Zhong et al., "Detection of urban flood inundation from traffic images using deep learning methods," Water Resources Management , vol. 38, pp. 287–301, 2024.
- [203] K. Lohumi et al., "Automatic detection of flood severity level from flood videos using deep learning models," in 2018 5th International Conference on Information and Communication Technologies for Disaster Management (ICT-DM). IEEE, 2018, pp. 1–7.
- [204] L. Lopez-Fuentes et al., "Multi-modal deep learning approach for flood detection." MediaEval, vol. 17, pp. 13–15, 2017.
- [205] M. Sandler et al., "Mobilenetv2: Inverted residuals and linear bottlenecks," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510–4520.
- [206] A. Krizhevsky et al., "Imagenet classification with deep convolutional neural networks," Advances in neural information processing systems , vol. 25, 2012.
- [207] C. Szegedy et al., "Going deeper with convolutions," in Proceedings of the IEEE conference on computer vision and pattern recognition , 2015, pp. 1–9.
- [208] S. Ren et al., "Faster r-cnn: Towards real-time object detection with region proposal networks," IEEE Trans. Pattern Anal. Mach. Intell. , vol. 39, pp. 1137–1149, 2016.
- [209] S. Woo et al., "Cbam: Convolutional block attention module," in Proceedings of the European conference on computer vision (ECCV) , 2018, pp. 3–19.
- [210] A. Van Den Oord et al., "Neural discrete representation learning," Advances in neural information processing systems, vol. 30, 2017.
- [211] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," arXiv preprint arXiv:2010.11929, 2020.
- [212] N. Humaira et al., "Dx-floodline: End-to-end deep explainable pipeline for real time flood scene object detection from multimedia images," IEEE Access, vol. 11, pp. 110 644–110 655, 2023.

![Image](artifacts/image_000007_eb12f1819572ba4c45b4e59425538b139b3c3f7e2bfac02768a54e48a9154315.png)

Ghazal Alinezhad Noghre is currently a Ph.D. candidate in Electrical and Computer Engineering at the University of North Carolina at Charlotte, Charlotte, North Carolina, United States. Her research focuses on artificial intelligence, machine learning, and computer vision, with a particular emphasis on the application of AI in real-world environments and the associated challenges.

![Image](artifacts/image_000008_1e65d2afaa5bbbe8f4c6c1128f0df2b159bc2297cbda019deebbb2b2dd007b86.png)

![Image](artifacts/image_000009_682b2232083f94475cf6f0c6a5590ea5083b5f9899fec791c9ab7a4fb3eb801c.png)

Armin Danesh Pazho is a Ph.D. candidate in Electrical and Computer Engineering at the University of North Carolina at Charlotte. His research focuses on artificial intelligence, machine learning, and computer vision, with emphasis on developing scalable AI solutions for practical applications. He has researched, designed, and developed novel AI/ML algorithms, systems, and datasets with deployment in real-world testbeds.

Hamed Tabkhi is an associate professor of Electrical and Computer Engineering at the University of North Carolina at Charlotte. His research focuses on advancing artificial intelligence and computer vision to solve real-world challenges through close collaboration with experts and community stakeholders. The National Science Foundation recognized Dr. Tabkhi's Smart and Connected Communities award as a program success story. His work has been featured by local news for its significant contributions to community-driven responsible AI solutions.

TABLE VIII LIST OF ABBREVIATIONS USED THROUGHOUT THE PAPER. THIS TABLE PROVIDES FULL FORMS FOR TECHNICAL TERMS COMMONLY REFERENCED IN THE CONTEXT OF VIDEO ANOMALY DETECTION (VAD).

| Abbreviation    | Full Form                                                         |
|-----------------|-------------------------------------------------------------------|
| VAD             | Video Anomaly Detection                                           |
| SOTA            | State-of-The-Art                                                  |
| AI              | Artificial Intelligence                                           |
| CNN             | Convolutional Neural Network                                      |
| VQ-VAE          | Vector Quantized Variational Autoencoder                          |
| ViT             | Vision Transformer                                                |
| GMM             | Gaussian Mixture Model                                            |
| GRU             | Gated Recurrent Unit                                              |
| LSTM            | Long Short-Term Memory                                            |
| RGB             | Red Green Blue (color video input)                                |
| MLP             | Multi-Layer Perceptron                                            |
| PD              | Parkinson’s Disease                                               |
| SVM             | Support Vector Machine                                            |
| GEI             | Gait Energy Image                                                 |
| ASD             | Autism Spectrum Disorde                                           |
| EEG             | Electroencephalography                                            |
| VEEG            | Video Electroencephalography                                      |
| MIL             | Multiple Instance Learning                                        |
| MSE             | Mean Squared Error                                                |
| GCN             | Graph Convolutional Networ                                        |
| VAE             | Variational Autoencoder                                           |
| GIN             | Graph Isomorphism Networ                                          |
| kNN             | k-Nearest Neighbors                                               |
| kDNN            | k-Nearest Distance Neural Network (DNN-based approximation of kNN |
| IoU             | Intersection-over-Union                                           |
| GAN             | Generative Adversarial Netwo                                      |

## IX. ABBREVIATIONS

This section provides a list of abbreviations and their corresponding full forms used throughout the survey. These terms are commonly referenced in the literature on Video Anomaly Detection (VAD). The purpose of this list is to assist readers with quick reference and improve the clarity and accessibility of the material presented.