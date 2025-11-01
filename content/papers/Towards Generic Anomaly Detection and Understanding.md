## Towards Generic Anomaly Detection and Understanding: Large-scale Visual-linguistic Model (GPT-4V) Takes the Lead

Yunkang Cao 1∗ , Xiaohao Xu 2∗ , Chen Sun 3∗ , Xiaonan Huang 2 and Weiming Shen 1

1 Huazhong University of Science and Technology 2 University of Michigan, Ann Arbor 3 University of Toronto

## Abstract

Anomaly detection is a crucial task across different domains and data types. However, existing anomaly detection models are often designed for specific domains and modalities. This study explores the use of GPT-4V(ision), a powerful visual-linguistic model, to address anomaly detection tasks in a generic manner. We investigate the application of GPT-4V in multi-modality, multi-domain anomaly detection tasks, including image, video, point cloud, and time series data, across multiple application areas, such as industrial, medical, logical, video, 3D anomaly detection, and localization tasks. To enhance GPT-4V's performance, we incorporate different kinds of additional cues such as class information, human expertise, and reference images as prompts. Based on our experiments, GPT-4V proves to be highly effective in detecting and explaining global and fine-grained semantic patterns in zero/one-shot anomaly detection. This enables accurate differentiation between normal and abnormal instances. Although we conducted extensive evaluations in this study, there is still room for future evaluation to further exploit GPT-4V's generic anomaly detection capacity from different aspects. These include exploring quantitative metrics, expanding evaluation benchmarks, incorporating multi-round interactions, and incorporating human feedback loops. Nevertheless, GPT-4V exhibits promising performance in generic anomaly detection and understanding, thus opening up a new avenue for anomaly detection.

All evaluation samples, including image and text prompts, will be available at https://github.com/caoyunkang/ GPT4V-for-Generic-Anomaly-Detection .

## Contents

| 1    | Introduction                                                     | Introduction                                                                                                                                                                                        |   5 |
|------|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| 1.1  | 1                                                                | Motivation and Overview                                                                                                                                                                             |   5 |
| 1.2  | Ou                                                               | r Approach: Prompting GPT-4V for Anomaly Detection .                                                                                                                                                |   5 |
|      | 1.2.1                                                            | Prompt Designs                                                                                                                                                                                      |   5 |
|      | 1.2.2                                                            | Evaluation Scope: Modalities and Domain                                                                                                                                                             |   6 |
| 1.3  | Lim                                                              | itations in Anomaly Detection Evaluation Based on GPT-4V                                                                                                                                            |   6 |
| 2    | Observations of GPT-4V on Multi-modal Multi-domain Anomaly Detec | Observations of GPT-4V on Multi-modal Multi-domain Anomaly Detec                                                                                                                                    |   7 |
| 2.1  | 2.1                                                              | GPT-4V can address multi-modality and multi-field anomaly detection tasks in zero/one-shot regime: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   7 |
| 2.2  | 2.2                                                              | GPT-4V can understand both global and fine-grained semantics for anomaly detection:                                                                                                                 |   7 |
| 2.3  | 2.3                                                              | GPT-4V can automatically reason for anomaly detection                                                                                                                                               |   8 |
| 2.4  | 4 G                                                              | GPT-4V can be enhanced with increasing prompts:                                                                                                                                                     |   8 |

∗ Authors contribute equally. Email: cyk\_hust@hust.edu.cn, xiaohaox@umich.edu, chrn.sun@mail.utoronto.ca, xiaonanh@umich.edu, wshen@ieee.org

| 2                                                      | 2.5                                                    | .5 GPT-4V can be constrained in real-world application but still promi   |   8 |
|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------|-----|
| 3 Industrial Image Anomaly Detection                   | 3 Industrial Image Anomaly Detection                   | 3 Industrial Image Anomaly Detection                                     |   8 |
| 3                                                      | 3.1                                                    | Task Introduction .                                                      |   8 |
| 3                                                      | 3.2                                                    | Testing philosophy                                                       |   8 |
| 3                                                      | 3.3                                                    | Case Demonstration                                                       |  10 |
| 4                                                      | dustrial Image Anomaly Localization                    | dustrial Image Anomaly Localization                                      |  10 |
| 4                                                      | Task Introd                                            | Task Introduction                                                        |  10 |
| 4.                                                     | 4.2                                                    | Testing philosophy                                                       |  10 |
| 4                                                      | 4.3                                                    | Case Demonstration                                                       |  10 |
| 5                                                      | oint Cloud Anomaly Detection                           | oint Cloud Anomaly Detection                                             |  10 |
| 5                                                      | 5.1                                                    | Task Introduction                                                        |  10 |
| 5.                                                     | 5.2                                                    | Testing philosophy                                                       |  11 |
| 5.                                                     | 5.3                                                    | Case Demonstration                                                       |  11 |
| 6 L                                                    | ogical Anomaly Detection                               | ogical Anomaly Detection                                                 |  11 |
| 6.                                                     | 6.1                                                    | Task Introduction .                                                      |  11 |
| 6.                                                     | 6.2                                                    | Testing philosophy                                                       |  11 |
| 6                                                      | 6.3                                                    | Case Demonstration                                                       |  12 |
| 6                                                      | Medical Image Anomaly Detection                        | Medical Image Anomaly Detection                                          |  12 |
|                                                        | 7.1                                                    | Task Introduction .                                                      |  12 |
| 7                                                      | 7.2                                                    | Testing philosophy                                                       |  12 |
| 7                                                      | 7.3                                                    | Case Demonstration                                                       |  13 |
| 8                                                      | Medical Image Anomaly Localization                     | Medical Image Anomaly Localization                                       |  13 |
| 8                                                      | 1 Task                                                 | Task Introduction                                                        |  13 |
| 8                                                      | 8.2                                                    | Testing philosophy                                                       |  13 |
| 8                                                      | 8.3                                                    | Case Demonstration                                                       |  13 |
| 8.3                                                    | 3 Case Demonstration                                   | 3 Case Demonstration                                                     |  14 |
| 9.                                                     | .1 Tas                                                 | Task Introduction                                                        |  14 |
| 9.2                                                    | 9.2                                                    | Testing philosophy                                                       |  14 |
| 9.3                                                    | 93                                                     | Case Demonstration                                                       |  14 |
| 0 Traffic Anomaly Detection                            | 0 Traffic Anomaly Detection                            | 0 Traffic Anomaly Detection                                              |  14 |
| 10.1 Task Introduction .                               | 10.1 Task Introduction .                               | 10.1 Task Introduction .                                                 |  14 |
| 10.2 Testing philosophy                                | 10.2 Testing philosophy                                | 10.2 Testing philosophy                                                  |  14 |
| 10.3 Case Demonstration .                              | 10.3 Case Demonstration .                              | 10.3 Case Demonstration .                                                |  15 |
| 10.3 Case Demonstration . .  Time Series Anomaly Dete | 10.3 Case Demonstration . .  Time Series Anomaly Dete | 10.3 Case Demonstration . .  Time Series Anomaly Dete                   |  15 |

| 11.1 T                  |   . . . 15 |
|-------------------------|------------|
| 11.2 Testing philosop   |         15 |
| 11.3 Case Demonstration |         15 |
| 12 Prospect             |         15 |
| 13 Conclusion           |         16 |

## List of Figures

|   1  | e Diagram of Evaluation GPT-4V on Multi-modality/fields Anomaly Detection. .    |   6 |
|------|---------------------------------------------------------------------------------|-----|
|   2  | Industrial Image Anomaly Detection: Case 1 .                                    |   9 |
|   3  | Industrial Image Anomaly Detection: Case 1                                      |  17 |
|   4  | Industrial Image Anomaly Detection: Case 2                                      |  18 |
|   5  | Industrial Image Anomaly Detection: Case 2                                      |  19 |
|   6  | Industrial Image Anomaly Detection: Case 3                                      |  20 |
|   7  | Industrial Image Anomaly Detection: Case 3                                      |  21 |
|   8  | Industrial Image Anomaly Localization: Case 1                                   |  22 |
|   9  | Industrial Image Anomaly Localization: Case 2                                   |  23 |
|  10  | Industrial Image Anomaly Localization: Case 3                                   |  24 |
|  11  | Point Cloud Anomaly Detection: Case 1 .                                         |  25 |
|  12  | Point Cloud Anomaly Detection: Case 1                                           |  26 |
|  13  | Point Cloud Anomaly Detection: Case 2                                           |  27 |
|  14  | Point Cloud Anomaly Detection: Case 2                                           |  28 |
|  15  | Point Cloud Anomaly Detection: Case                                             |  29 |
|  16  | Point Cloud Anomaly Detection: Ca                                               |  30 |
|  17  | Logical Anomaly Detection: Case 1 .                                             |  31 |
|  18  | Logical Anomaly Detection: Case 2 .                                             |  32 |
|  19  | Logical Anomaly Detection: Case 3 .                                             |  33 |
|  20  | Logical Anomaly Detection: Case 4                                               |  34 |
|  21  | Medical Anomaly Detection: Case 1                                               |  35 |
|  22  | Medical Anomaly Detection: Case 1                                               |  36 |
|  23  | Medical Anomaly Detection: Case 2                                               |  37 |
|  24  | Medical Anomaly Detection: Case 2                                               |  38 |
|  25  | Medical Anomaly Detection: Case 3                                               |  39 |
|  26  | Medical Anomaly Detection: Case 3                                               |  40 |
|  27  | Medical Anomaly Detection: Case 4                                               |  41 |
|  28  | Medical Anomaly Detection: Case 4                                               |  42 |
|  29  | Medical Anomaly Localization: Case 1                                            |  43 |
|  30  | Medical Anomaly Localization: Case 2                                            |  44 |
|  31  | Medical Anomaly Localization: Case 3                                            |  45 |
|  32  | Medical Anomaly Localization: Case 4                                            |  46 |
|  33  | Pedestrian Anomaly Detection                                                    |  47 |
|  34  | Traffic Anomaly Detection: Case 1                                               |  48 |
|  35  | Traffic Anomaly Detection: Case 2                                               |  49 |
|  36  | Time Series Anomaly Detection: Case                                             |  50 |
|  37  | Time Series Anomaly Detection: Case 2                                           |  51 |

## 1 Introduction

## 1.1 Motivation and Overview

Anomaly detection [20 , 19 , 72 , 10 , 78] involves identifying data patterns or data points that significantly deviate from normality. These anomalies or outliers are rare, unusual, or inconsistent data points that deviate from the majority of the data. The primary objective of anomaly detection is to automatically detect and pinpoint these irregularities, which may signify errors, fraud, unusual events, or other noteworthy phenomena, facilitating further investigation or necessary action. Anomaly detection techniques have been widely employed in diverse domains, such as industrial inspection [29 , 98], medical diagonisis [107], video surveillance [84], fraud detection [30] and many other areas where identifying unusual instances is crucial.

Despite the existence of numerous techniques [14 , 3 , 69 , 41 , 38 , 79 , 110 , 16 , 103] for anomaly detection, many existing approaches predominantly rely on methods that describe the normal data distribution. They often overlook high-level perception and primarily treat it as a low-level task. However, practical applications of anomaly detection frequently necessitate a more comprehensive, high-level understanding of the data. Achieving this understanding entails at least three crucial steps:

1. Understanding the Data Types and Categories: The first step involves a thorough comprehension of the data types and categories present in the dataset. Data can take various forms, including images, videos, point clouds, time-series data, etc. Each data type may require specific methods and considerations for anomaly detection. Furthermore, different categories may have distinct definitions of normal states.
2. Determining Standards for Normal States: After obtaining the data types and categories, it would be feasible to further reason the standards for normal states, which requires a high-level understanding of the data.
3. Evaluating Data Conformance: The final step is to assess whether the provided data conforms to the established standards for normality. Any deviation from these standards can be categorized as an anomaly.

Recent advancements in large multimodal models (LMMs) [25 , 4 , 36 , 113 , 58 , 27 , 52] have shown robust reasoning capacity [55 , 57] and created new opportunities for improving anomaly detection. LMMs are typically trained on extensive multimodal datasets [80], enabling them to effectively analyze various data types, including natural language and visual information. They hold the potential to address the challenges associated with high-level anomaly detection [37 , 17 , 22 , 112].

Moreover, OpenAI recently introduced GPT-4V(ision) [101], a state-of-the-art LMM that has exhibited remarkable performance across various practical applications. However, it remains uncertain whether GPT-4V can also exhibit robust capabilities for anomaly detection. The objective of this study is to bridge this knowledge gap by assessing the anomaly detection capabilities of GPT-4V.

## 1.2 Our Approach: Prompting GPT-4V for Anomaly Detection

## 1.2.1 Prompt Designs

The design of prompts plays a crucial role in effectively directing GPT-4V's attention toward the specific aspects of the anomaly detection task. In this study, we primarily consider four types of prompts:

1. Task Information Prompt: To prompt GPT-4V effectively for anomaly detection, it is essential to provide clear task information. This study formulates the prompt as follows: "Please determine whether the image contains anomalies or outlier points."
2. Class Information Prompt: The understanding of data types and categories is critical. In cases where GPT-4V may struggle to recognize the data class, explicit class information may be provided. For instance, "Please determine whether the image, which is related to the {CLS}, contains anomalies or defects."

Figure 1 | Comprehensive Evaluation of GPT-4V for Multi-modality Multi-task Anomaly Detection In this study, we conduct a thorough evaluation of GPT-4V in the context of multi-modality anomaly detection. We consider four modalities: image, video, point cloud, and time series, and explore nine specific tasks, including industrial image anomaly detection/localization, point cloud anomaly detection, medical image anomaly detection/localization, logical anomaly detection, pedestrian anomaly detection, traffic anomaly detection, and time series anomaly detection. Our evaluation encompasses a diverse range of 15 datasets.

![Image](artifacts/image_000000_74890c9f7bf1852782d210b40316c7af32dba7f347767588eb667d37649d6883.png)

3. Normal Standard Prompt: GPT-4V may encounter difficulties in answering questions related to determining normal standards, and sometimes the standards even can not be examined without human expertise. Hence, this study also explicitly provides the normal standards. For example, normal standards for the breakfast box in MVTec-LOCO [7] could be expressed as follows: "1. It should contain two oranges, one peach, and some cereal, nuts, and banana slices; 2. The fruit should be on the left side of the lunchbox, the cereal on the upper right, and the nuts and banana slices on the lower right of the lunchbox."
4. Reference Image Prompt: To ensure better alignment between normal standards and images, a normal reference image is provided alongside language prompts. For example, "The first image is normal. Please determine whether the second image contains anomalies or defects."

The study aims to explore how the use of these prompts, either individually or in different combinations depending on certain cases, impacts GPT-4V's capacity for anomaly detection.

## 1.2.2 Evaluation Scope: Modalities and Domains

Extensive evaluations are conducted in this study to assess the capabilities of GPT-4V in anomaly detection, as Fig. 1 shows. From the perspective of modalities, we evaluate image (Section 3 , 4 , 6 , 7 , 8), point cloud (Section 5), video (Section 9 , 10), and time series (Section 11). From the perspective of fields, industrial inspection (Section 3 , 4 , 6 , 5), medical diagnosis (Section 7 , 8), and video surveillance (Section 9 , 10) are evaluated. To the best of our knowledge, this is the first study to investigate such a wide range of modalities and fields for anomaly detection.

## 1.3 Limitations in Anomaly Detection Evaluation Based on GPT-4V

The analysis of this study is subject to certain limitations:

1. Predominance of Qualitative Results: The analysis primarily relies on qualitative assessment,

lacking quantitative metrics that could offer a more objective evaluation of the model's performance in anomaly detection. Incorporating quantitative measures would provide a more robust basis for assessment.

2. Scope of Evaluated Cases: The evaluation is confined to a limited scope of cases or scenarios. This narrow focus may not fully capture the diverse challenges encountered in real-world anomaly detection tasks. Expanding the range of evaluated cases would yield a more comprehensive understanding of the model's capabilities.
3. Single Interaction Evaluation: The study mainly concentrates on a single-round conversation. In contrast, multi-round conversations, as observed in the in-context learning capacity of GPT-4V [101], can stimulate deeper interaction. The single-round conversation approach restricts the depth of interaction and may constrain the model's comprehension and its effectiveness in responding to anomaly detection tasks. Exploring multi-round interactions could reveal a more nuanced perspective of the model's performance.

## 2 Observations of GPT-4V on Multi-modal Multi-domain Anomaly Detection

Following a thorough evaluation of GPT-4V's performance across various multi-modality and multi-field anomaly detection tasks, it becomes apparent that GPT-4V possesses robust anomaly detection capabilities. More precisely, GPT-4V consistently excels in addressing the three previously mentioned challenges: comprehending image context, discerning normal standards, and effectively comparing the provided image against these standards. In addition to these fundamental findings, our assessments have yielded valuable insights.

## 2.1 GPT-4V can address multi-modality and multi-field anomaly detection tasks in zero/one-shot regime:

Anomaly detection for multi-modality: GPT-4V's ability to handle diverse data modalities is demonstrated by its consistent performance across various domains. For instance, it exhibits proficiency in identifying anomalies in images, point clouds, X-rays, etc., underscoring its adaptability to multi-modal tasks. This versatility allows it to transcend the limitations of single-modal anomaly detectors.

Anomaly detection for multi-field: GPT-4V's performance across multiple fields, including industrial, medical, pedestrian, traffic, and time series anomaly detection, showcases its ability to seamlessly adapt to the distinct characteristics of each domain. Its consistent results affirm its broad applicability and versatility, making it a valuable tool for anomaly detection in a variety of real-world contexts.

Anomaly detection in zero/one-shot regime: GPT-4V's evaluation in both zero-shot and one-shot settings highlights its adaptability to different inference scenarios. In the absence of reference images, the model effectively relies on language prompts to detect anomalies. However, when provided with normal reference images, its anomaly detection accuracy is further enhanced. This flexibility enables GPT-4V to cater to a wide range of anomaly detection applications, whether with or without prior knowledge.

## 2.2 GPT-4V can understand both global and fine-grained semantics for anomaly detection:

GPT-4V's understanding of global semantics: GPT-4V's capacity to comprehend global semantics is demonstrated in its ability to recognize overarching abnormal patterns or behaviors. For example, in traffic anomaly detection, it can discern the distinction between typical traffic flow and irregular events, providing a holistic interpretation of the data. This global understanding makes it well-suited for identifying anomalies that deviate from expected norms in a broader context.

GPT-4V's understanding of fine-grained semantics: GPT-4V's fine-grained anomaly detection capabilities shine in cases where it not only detects anomalies but also precisely localizes them within complex data. For instance, in industrial image anomaly detection, it can pinpoint intricate details like slightly tilted wicks on candles or minor scratches or residues around the top rim of the bottle. This fine-grained understanding enhances its ability to detect subtle anomalies within complex data, contributing to its overall effectiveness.

## 2.3 GPT-4V can automatically reason for anomaly detection:

The model's strength in automatically reasoning the given complex normal standards and generating explanations for detected anomalies is a valuable feature. In logical anomaly detection, for example, GPT-4V excels at dissecting complex rules and providing detailed analyses of why an image deviates from the expected standards. This inherent reasoning ability adds a layer of interpretability to its anomaly detection results, making it a valuable tool for understanding and addressing irregularities in various domains.

## 2.4 GPT-4V can be enhanced with increasing prompts:

The results of the evaluation highlight the positive impact of additional prompts on GPT-4V's anomaly detection performance. The model's response to class information, human expertise, and reference images suggests that providing it with more context and information significantly improves its ability to detect anomalies accurately. This feature allows users to fine-tune and enhance the model's performance by providing relevant and supplementary information.

## 2.5 GPT-4V can be constrained in real-world application but still promising:

From the cases we test, we find there are still several gaps for GPT4V models to be applied in real world anomaly detection. For example, GPT-4V may face challenges in handling highly complex scenarios for industrial application. Ethical constraints in the medical field also make it conservative and hesitate to give confident answer. But we believe it remains promising in a wide range of anomaly detection tasks. To address these challenges effectively, further enhancements, specialized fine-tuning, or complementary techniques may be required. GPT-4V's potential for anomaly detection is evident, and ongoing research may continue to unlock its capabilities in even more complex scenarios.

## 3 Industrial Image Anomaly Detection

## 3.1 Task Introduction

Industrial image anomaly detection is a critical component of manufacturing processes aimed at upholding product quality [6 , 98 , 14]. Following the establishment of the MVTec AD dataset [6], various methods [15 , 45 , 22 , 17 , 15 , 46 , 92] have thrived in this field. These methods focus on determining whether testing images contain anomalies, typically represented as local structural variants. Early methods [91 , 95 , 102 , 13 , 54 , 94] concentrated on developing specific models for given categories, while recent approaches [45 , 22 , 17 , 112] target a more general but challenging solution, i.e., developing a unified model for arbitrary product categories, which usually performs in few-shot [99 , 40] or even zero-shot [45 , 17 , 22] regime. As highlighted in [101], GPT-4V, equipped with extensive world knowledge, presents a promising solution for arbitrary category inspection.

## 3.2 Testing philosophy

Different prompts [101 , 56] could lead to different responses from GPT-4V. We aim to investigate the influence of different information on prompting GPT-4V for industrial anomaly detection. Following the previously discussed problems, this study further develops three prompts, a) class information: the names of the desired inspecting products, such as "bottle" and "candle", b) human expertise: the normal appearance and potential abnormal states and express them in languages, e.g., "Normally, the image given should show a clean and well-structured printed circuit board (PCB) with clear traces, soldered components, and distinct labels. It may have defects such as bent pins, cold solder joints, missing components, or smudged labels", c) reference image: normal reference image to provide GPT-4V a better understanding of normality. We propose to evaluate GPT-4V in either a zero-shot setting, with only language prompts, or a one-shot setting, with one reference image provided along with the language prompts. For each setting, we test three different variants: a) a naive prompt like "Please determine whether the image contains anomalies or defects," b) with class information, and c) with human expertise.

![Image](artifacts/image_000001_61000dac9995d4089db7205d41469131148c807b563ba38dc433f5f3d7738727.png)

Figure 2 | Industrial Image Anomaly Detection: Case 1, zero-shot, the Bottle category of MVTec AD [6] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

## 3.3 Case Demonstration

Fig. 2 , 3 , 4 , 5 , 6 , 7 qualitatively demonstrate the effectiveness of GPT-4V for industrial image anomaly detection. Even with a simple language prompt, GPT-4V effectively identifies anomalies in examined bottle and candle images, showcasing its capacity and versatility. Moreover, GPT-4V excels not only in detecting desired anomalies but also in identifying fine-grained structural anomalies. As evident in Fig. 4, GPT-4V noticed a slightly tilted wick on the bottom left candle, demonstrating its nuanced understanding. In complex cases like Fig. 6, GPT-4V recognizes the PCB in images and provides in-depth reasoning about anomalies, such as examining the proper seating of the ultrasonic sensor. However, GPT-4V overlooks the bent pin, resulting in an incorrect result. Nevertheless, GPT-4V showcases a strong grasp of image context and category-specific anomaly understanding.

## 4 Industrial Image Anomaly Localization

## 4.1 Task Introduction

Industrial image anomaly localization entails a more intricate process than mere image anomaly detection [76 , 93 , 12 , 13 , 65]. It goes beyond recognizing the abnormality within an image and extends to precisely identifying the location of these anomalies. While GPT-4V has exhibited localization capabilities in various domains [101 , 97 , 100], its potential for image anomaly localization warrants further exploration.

Regrettably, GPT-4V does not currently have the capability to directly produce prediction masks. Some methods have attempted to leverage GPT-4V by prompting it to provide bounding boxes [101 , 97]. However, this approach appears to be imprecise and poses challenges for GPT-4V. In contrast, the approach presented by SoM [100] involves utilizing SAM [50] to generate visual prompts [81 , 50], which are presented in numbered markers. This visual prompting technique shifts the localization task from a pixel-level mask prediction task to a mask-level classification task, effectively reducing the associated complexities and increasing localization precision.

## 4.2 Testing philosophy

To harness the fine-grained localization capability of GPT-4V, we adopt the approach outlined in SoM [100]. This involves generating a set of image-mask pairs for prompting GPT-4V. In addition to the image-mask pairs, we employ a straightforward language prompt that instructs the model, as follows: "The first image needs to be inspected. The second image contains its corresponding marks. Please determine whether the image contains anomalies or defects. If yes, give a specific reason".

## 4.3 Case Demonstration

Fig. 8 , 9, and 10 provide a visual representation of GPT-4V's performance in industrial anomaly localization. These illustrations clearly illustrate GPT-4V's ability to accurately identify the second mask in Fig. 8 as a twisted wire and the second mask in Fig. 9 as holes. These results serve as compelling evidence of GPT-4V's proficiency in localizing anomalies when guided by visual prompts.

It is important to acknowledge that GPT-4V does exhibit certain limitations when confronted with more complex scenarios, as evidenced in Fig. 10. However, the combination of visual prompting techniques and GPT-4V remains a promising approach for industrial anomaly localization.

## 5 Point Cloud Anomaly Detection

## 5.1 Task Introduction

Geometrical information, as discussed in references such as PAD [111], Real3D [59], and MVTec-3D [8], holds a crucial role in fields like industrial anomaly detection, especially when dealing with categories lacking textual information. Recently, MVTec 3D [8] and Real3D [59] have recognized the growing need for such information and have introduced a point cloud anomaly detection task. This task focuses on the identification of anomalies

within the provided point clouds [32].

It is important to note that the success achieved in industrial image anomaly detection is not fully mirrored in point cloud anomaly detection. This disparity is primarily attributed to the reliance of industrial image anomaly detection on robust pre-trained networks [12 , 75 , 39]. Conversely, due to the scarcity of extensive point cloud data, the capabilities of pre-trained networks for point clouds currently fall short, leading to suboptimal performance for some methods [96 , 21 , 9 , 77].

In contrast, CPMF [16] proposes a novel approach by transforming point clouds into depth images, thereby opening up the possibility of leveraging image-based foundation models for point cloud anomaly detection. This innovative method has shown the potential to deliver significantly improved results in point cloud anomaly detection.

## 5.2 Testing philosophy

To employ GPT-4V in the context of point cloud anomaly detection, we adopt the methodology presented in CPMF [16] to transform point clouds into multi-view depth images. In our evaluation, we adhere to the principles commonly used in industrial image anomaly detection, specifically the zero/one-shot approach, with the inclusion of three distinct variations of language prompts.

## 5.3 Case Demonstration

Fig. 11 , 12 , 13 , 14 , 15 , 16 provide a visual representation of the performance of GPT-4V in point cloud anomaly detection. These illustrations serve to qualitatively illustrate the proficiency of GPT-4V in comprehending multi-modality data.

Specifically, GPT-4V demonstrates its capability to accurately identify the presence of a small protrusion or bump on the top left part of the torus in the bagel (Fig. 11). Moreover, the introduction of additional information, such as class information and human expertise, enhances the performance of GPT-4V in point cloud anomaly detection, allowing it to effectively detect anomalies in the rope (Fig. 15 and 16).

However, it is noteworthy that GPT-4V may occasionally misidentify artificially introduced elements during the rendering process as anomalies, as observed in Fig. 14. It is possible that improvements in rendering quality could further enhance the capacity of GPT-4V in this context.

## 6 Logical Anomaly Detection

## 6.1 Task Introduction

In addition to structural anomalies, there exists another type of anomaly, named logical anomalies [7]. Logical anomalies generally refer to incorrect combinations of components, commonly encountered in the context of anomaly detection in assemblies. For instance, a screw bag should contain matched screws, nuts, and washers. This necessitates that the model is capable of understanding fine-grained information in images and determining attributes of the components within the image, such as component type, length, color, quantity, and so forth. This places higher demands on the model. Existing logical anomaly detection methods [63 , 103 , 5 , 106] typically relied on solely visual context and have achieved promising detection performance. However, these approaches do not genuinely comprehend the content of images; instead, they rely on global-local correspondences [98] for logical anomaly detection. This does not effectively address logical anomaly detection. In contrast, GPT-4V possesses robust image understanding capabilities, allowing for a better comprehension of image content. By providing predefined normal rules manually, GPT-4V might be capable of determining whether an image adheres to normal rules, thereby enabling a more rational approach to logical anomaly detection.

## 6.2 Testing philosophy

To ensure an effective assessment of testing images, it is crucial to provide clear guidelines defining the expected normal state for GPT-4V. This enables GPT-4V to evaluate the conformity of testing images with

the established standards, relying on an analysis of image content in relation to these norms. Consequently, our approach involves presenting GPT-4V with both testing images and descriptive language articulating the expected normal standards. However, it is worth noting that GPT-4V might encounter difficulties in comprehending the nuances of normal standards when presented with language alone. To enhance its understanding and alignment of normal standards with the context of normal images, we propose the inclusion of a reference image illustrating the desired normal state. Therefore, our experimental design encompasses both zero-shot and one-shot settings to assess the effectiveness of this approach.

## 6.3 Case Demonstration

The evaluation results, as depicted in Fig. 17 , 18 , 19 , 20, unequivocally highlight the robust image comprehension and logical reasoning capabilities of GPT-4V. For instance, in Fig. 17, GPT-4V demonstrates its proficiency in interpreting intricate standards, encompassing criteria such as the presence of "1. It should contain two oranges, one peach, and some cereal, nuts, and banana slices; 2. The fruit should be on the left side of the lunch box, the cereal on the upper right, and the nuts and banana slices on the lower right of the lunch box". GPT-4V adeptly breaks down this complex task into subcomponents, identifying and localizing the various items before calculating their quantities and positions. Ultimately, GPT-4V accurately concludes that the provided breakfast box does not adhere to the stipulated standards.

Moreover, visual references play a pivotal role in enhancing GPT-4V's performance. In Fig. 18, without the aid of a visual reference, GPT-4V erroneously classifies the juice bottle as a normal one. However, when presented with a referenced image, GPT-4V effectively comprehends the rule "2. To prevent bottle explosions, ensure the juice is filled to about 3cm below the bottle's opening" and delivers a correct analysis.

Nonetheless, GPT-4V may encounter challenges in scenarios where its ability to contextualize images is constrained. Notably, GPT-4V fails to detect a broken cable in Fig. 19 and inaccurately quantifies washers in Fig. 20. The limitations of GPT-4V, particularly in matters of fine-grained details like counting, have been addressed in prior research [101]. Furthermore, it is worth noting that multi-round conversations and specific language prompts can significantly impact GPT-4V's performance in such cases.

## 7 Medical Image Anomaly Detection

## 7.1 Task Introduction

Anomaly detection, also known as outlier detection, is a pivotal task in the domain of medical imaging, aimed at identifying abnormal patterns that do not conform to expected behavior[31]. These abnormalities or anomalies could be indicative of a wide range of medical conditions or diseases[citation]. The primary goal of anomaly detection is to accurately discern these irregularities from a plethora of medical imaging data, thereby aiding in early diagnosis and effective treatment planning. Current medical anomaly detection methods can be categorized into reconstruction-based methods [23] [35] [90], GAN-based [61], self-supervised methods [82] [88] [87] and pre-train methods [75] [28] [60] [62] Although these methods have achieved great improvements, a unified anomaly detection model across different diseases and modalities still remains an unsolved challenge. As highlighted in [71] and [97], GPT-4V, equipped with numerous multi-modal knowledge, shows promising future in enhancing the performance of anomaly detection tasks in various medical imaging modalities.

## 7.2 Testing philosophy

We aim to investigate the generalization abilities of GPT-4V on medical anomaly detection. Thus medical images on across different diseases and modalities are used, including Head MRI, Head CT, Retinal OCT, Chest X-ray and so on. For the text prompt, we also take the previous multi-step prompt to test its zero-shot and one-shot abilities. There are generally three types of prompts, a)general medical information, the disease and modalities of the medical images, such as "Chest X-ray Image" or "Head CT Image" b)human expertise, based on the general medical information, we further give the possible disease name in the medical image, e.g."The image should be classified as normal or hemorrhage", c) reference image: normal reference image to provide GPT-4V a better understanding of nomrality.

We propose to evaluate GPT-4V in either a zero-shot setting, with only language prompts, or a one-shot setting, with one reference image provided along with the language prompts. For each setting, we test three different variants: a) a naive prompt like "Please determine whether the image contains anomalies" b)general medical information, and c) with human expertise.

## 7.3 Case Demonstration

Fig.21 , 23 , 25 and 27 show the GPT-4V's zero-shot inference ability. GPT-4V is capable of automatically recognizing medical image modalities and anatomical structures, even without general medical information prompts. The superior image caption ability enables GPT-4V to describe the spatial and textural anomalies in the image. However, due to ethical restrictions, the GPT-4V model tends to give conservative answers when lack of sufficient information. The introduction of both general medical information and human expertise successfully leads GPT-4V to generate more concrete and accurate answers, as shown in Fig 21 , 23 and 25. However, GPT-4V fails to recognize anomalies in Fig 27, even with enough information provided. The abnormal area is not obvious in the image, so it turns out that it has high requirements for the medical image quality. When a visual reference is added, the GPT-4V's image caption ability successfully describe the difference between normal and abnormal images, which is shown in Fig 22 , 24 26 and 28 .

## 8 Medical Image Anomaly Localization

## 8.1 Task Introduction

Following the detection of medical anomaly, the subsequent critical task is anomaly localization, which entails pinpointing the exact spatial location of the identified anomaly within the medical image [88] [104]. Accurate localization is imperative for clinicians to understand the extent and nature of the pathology, which in turn informs the course of clinical intervention. However, the real-world clinical scenario, such as tumor anomaly localization, is more complex, where either normal or abmoral cases have multiple types of tumors. Establishing a direct relationship between image pixels and excessive semantics (types of tumors) is diffcult for real world medical image anomaly localization. Several methods, including self-supervised based method [88] and cluster-based method [104] have been proposed to deal with the medical image anomaly localization task. Inspired by [100], we would like to examine the localization ability of GPT-4V model, under the visual prompts.

## 8.2 Testing philosophy

To test the GPT-4V's ability on medical image localization, we utilize several diseases categories and modalities, including abdominal CT image, endoscopy image, head MRI image and skin lesion image. Both diseased area and manually synthetic abnormal are taken into consideration to test its robustness. The visual prompts proposed by [100] are also used to harness the fine-grained localization abilities of GPT-4V, including a set of image-mask pairs and corresponding index numbers to each mask. Thus, the input images are the raw images with the augmented one with masks and numbers. We also adopt a straightforward text prompt to introduce the relationship between the two input images, as follows: "The first image needs to be inspected. The second image contains its corresponding marks. Please determine whether the image contains anomalies or defects. If yes, give a specific reason"

## 8.3 Case Demonstration

The qualitative results are shown in Fig 29 30 31 and 32. Under the instruction of visual prompts in the images, the GPT-4V tends to learn and caption the areas around the marks. For easily recognized and located cases, such as Fig 30 31 and 32, GPT-4V can clearly tell the difference between the anomaly areas and backgrounds. But GPT-4V fails in Fig 29, a synthetic case where the region-of-interest shares a similar texture and shape with the background. This indicates that this model still needs to improve its detection and localization abilities under adversarial attack and complex backgrounds.

## 9 Pedestrian Anomaly Detection

## 9.1 Task Introduction

Pedestrian anomaly detection, a subset of video anomaly detection, is dedicated to recognizing irregular activities within pedestrian interactions captured in video streams. Traditional methodologies, as referenced by various studies [1 , 69 , 33 , 109 , 64 , 86 , 105 , 44], primarily rely on rule-based approaches and manually engineered features. In recent times, there has been a noticeable shift towards the adoption of deep learning techniques [38 , 24 , 66 , 74 , 73 , 53 , 42 , 43 , 41] for pedestrian anomaly detection. The complexity of pedestrian anomaly detection arises from the need to accurately identify abnormal behaviors within the context of diverse and dynamic pedestrian interactions. This is further compounded by the varying environmental conditions in which these interactions take place. To ensure precise analysis, a substantial contextual understanding is essential. While existing methods have demonstrated promising performance in pedestrian anomaly detection, it is worth considering that GPT-4V, with its advanced contextual comprehension capabilities, has the potential to significantly enhance the performance of this task.

## 9.2 Testing philosophy

We utilize the GPT-4V model, which currently only accepts image format visual input, for pedestrian anomaly detection. To prompt the model, we select two images from the video dataset. In addition to the image prompt, we include a simple text prompt asking the model to determine if the video frames contain anomalies or outlier points and provide a specific reason if so.

## 9.3 Case Demonstration

In Fig. 33, we illustrate a scenario (from UCF-Crime datadet [85]) where a pedestrian aggresses another on the road. The GPT-4V model recognizes the aggressive behavior as an anomaly when compared to typical interactions. Additionally, it suggests caution due to the "LiveLeak" watermark, implying a need for further analysis with sufficient contextual information before drawing conclusions. The model's adeptness at discerning aggressive behavior, even in the absence of technical anomalies, demonstrates its potential to identify social anomalies within visual data.

## 10 Traffic Anomaly Detection

## 10.1 Task Introduction

Traffic anomaly detection primarily aims at identifying the commencement and conclusion of abnormal events, with lesser emphasis on spatial localization. Various methodologies [38 , 70 , 24 , 67 , 68 , 35 , 35] have been devised to model normalcy and discern regular patterns in video frames. The prevailing challenge for anomaly detection in traffic scenarios is the development of robust algorithms that can effectively differentiate between normal and abnormal vehicles and driving behaviors, thereby ensuring the safety and reliability of the autonomous vehicle system. Integrating GPT4v into traffic anomaly detection promises to refine the precision and speed of current systems. GPT4v, which has the ability to conduct high-level understanding, is adept at parsing the intricacies of traffic data, thereby sharpening the discrepancy between normal variations and true anomalies. This precision is critical for developing real-time monitoring systems that deliver accurate alerts while minimizing false positives.

## 10.2 Testing philosophy

We employ GPT-4V for traffic anomaly detection, which, as of now, only accepts visual input in image format. To engage the model, we select a representative image from the traffic scene, accompanied by a succinct text prompt. This prompt requests the model to ascertain whether the image frames harbor anomalies or outlier points, and if found, to elucidate the specific reasons for such irregularities.

## 10.3 Case Demonstration

As depicted in Fig. 34 and 35, by scrutinizing the spatial-temporal dynamics within the traffic scenes from a traffic anomaly detection dataset [48], GPT-4V proficiently differentiates between standard traffic flow and anomalous events. Beyond merely identifying outliers in traffic patterns, the model extends its utility by offering insightful elucidations concerning the abnormal nature of the scenarios. For instance, in Fig.34 , the model effectively explicates an abnormal vehicular maneuver that collides with the roadside barrier and deviates from typical driving behavior. Harnessing its deep comprehension of the underlying patterns and relationships within the traffic data, the model employs interpretable techniques to unravel the factors contributing to the anomaly, thereby providing a nuanced understanding that could be pivotal for enhancing the safety and reliability of autonomous driving systems.

## 11 Time Series Anomaly Detection

## 11.1 Task Introduction

Time series anomaly detection refers to the task of identifying unusual or abnormal patterns, events, or behaviors in sequential data over time, that deviate significantly from the expected or normal behavior. Time series anomaly detection models can be categorized as supervised or unsupervised algorithms. Supervised methods perform well when anomaly labels are available, such as AutoEncoder [79] and RobustTAD[34]. Unsupervised algorithms are suitable when obtaining anomaly labels is challenging. This has led to the development of new unsupervised methods, including DAGMM [115] and OmniAnomaly [83]. Unsupervised deep learning methods excel in time series anomaly detection, leveraging representation learning and a reconstruction approach to accurately identify anomalies without the need for labeled data [110 , 47 , 108].

## 11.2 Testing philosophy

To exploit GPT-4V for time series anomaly detection, we plot time series into images and then deliver the testing data to GPT-4V. Specifically, we select two instances [2 , 89] along with a simple text prompt asking the model to determine if the image contains anomalies or outlier points and provide a specific reason if so.

## 11.3 Case Demonstration

As illustrated in Fig. 36 and 37, by examining the temporal dependencies and trends within the time series, GPT-4V adeptly differentiates between normal fluctuations and anomalous behavior. Beyond merely detecting outliers in the time series curves, the model extends its utility by offering insightful explanations regarding the abnormal nature of the data. For instance, in Fig. 37, the model effectively elucidates the abnormal peak in the time series. Drawing upon its profound understanding of the underlying patterns and relationships within the data, the model employs interpretability techniques to illuminate the factors contributing to the anomaly.

## 12 Prospect

The future evaluation and utilization of GPT-4V for anomaly detection hold significant promise in addressing complex challenges across various domains. As a versatile language model, GPT-4V demonstrates its potential in anomaly detection, and the following prospects aim to refine its capabilities, foster integration, and elevate its performance.

1. Quantitative Analysis: Incorporating quantitative metrics, such as Precision, Recall, and F1-score, alongside AUC-ROC and MAP, in future evaluations will provide a more comprehensive understanding of GPT-4V's anomaly detection performance. This quantification will empower a more objective assessment of the model's capabilities and its adaptation to diverse anomaly detection tasks.
2. Expanding Evaluation Scope: Expanding the scope to include real-world challenges, such as varying lighting conditions and occlusions in image-based anomaly detection, and different types of anomalies in time-series data, offers a more realistic view of GPT-4V's adaptability and limitations. The inclusion of synthetic and real-world anomalies adds depth to the evaluation process.

3. Multi-round Interaction Evaluation: The potential of multi-round conversations for GPT-4V's iterative learning and adaptation to feedback provides a dynamic framework for enhancing its performance in anomaly detection. It is a promising avenue for scenarios where ongoing refinement is crucial, such as cybersecurity.
4. Incorporation of Human Feedback: Utilizing human feedback loops presents the opportunity for domain experts to refine GPT-4V's understanding of complex or nuanced anomalies. The collaboration between the model and experts promises to address real-world challenges effectively.
5. Integration of Auxiliary Data: Exploring the impact of integrating auxiliary data, such as additional sensor readings or metadata, is instrumental in enhancing GPT-4V's understanding and accuracy in identifying anomalies across various domains. This comprehensive approach aligns with real-world data scenarios.
6. Comparison with Specialized Models: Comparative evaluations against specialized anomaly detection models are essential to identify the specific strengths and weaknesses of GPT-4V. These assessments will clarify the domains and use cases where GPT-4V's versatility excels or where specialized models remain superior.
7. Real-Time Performance Assessment: Evaluating GPT-4V's real-time performance is crucial for applications requiring rapid anomaly detection. This prospect ensures the model's suitability for time-critical or online anomaly detection tasks.
8. Transfer Learning Evaluation: Assessing the effectiveness of transfer learning in fine-tuning GPT-4V for specific anomaly detection tasks can pave the way for broader generalization. It enhances the model's adaptability in diverse anomaly detection scenarios.
9. Hybrid Model Development: The development of hybrid models combining GPT-4V with other machine learning or deep learning approaches offers an innovative approach to address anomaly detection challenges. These hybrids aim to leverage GPT-4V's linguistic capabilities while enhancing its performance in specialized scenarios.

In summation, these prospects set the stage for a comprehensive and multifaceted exploration of GPT-4V's anomaly detection potential. By combining quantitative metrics, real-world challenges, human feedback, auxiliary data integration, comparative assessments, and real-time capabilities, we can unlock the full scope of GPT-4V's utility in addressing anomalies across diverse fields. The journey towards improved anomaly detection with GPT-4V is one of collaboration, adaptation, and innovation, promising exciting developments in the years to come.

## 13 Conclusion

In conclusion, the assessment of GPT-4V's capabilities in anomaly detection signifies a notable advancement in the realm of versatile and adaptable AI models. GPT-4V demonstrates exceptional proficiency in identifying anomalies across diverse modalities and fields, offering both comprehensive and nuanced semantic comprehension. Its ability to deduce anomalies and its responsiveness to an expanding array of prompts underscore its versatility and potential. Nevertheless, like any technology, there remains room for further enhancement, particularly in intricate and subtle scenarios.

The opportunities delineated in this evaluation propose promising avenues for future research and development. The inclusion of quantitative metrics, broadening the spectrum of evaluations, embracing human input, and integrating supplementary data all contribute to augmenting the performance of GPT-4V. Comparative assessments against specialized models and the exploration of hybrid models further enrich the landscape of anomaly detection. Real-time assessment and the incorporation of transfer learning hold the promise of addressing time-sensitive situations and generalizing anomaly detection across diverse domains.

As we embark on this journey to unlock the full potential of GPT-4V, collaboration, adaptability, and innovation will serve as the foundational pillars of our success. The evaluation and utilization of GPT-4V for anomaly detection do not merely signify an exploration of technology but also serve as a testament to the ongoing evolution of AI and its transformative impact on real-world applications. Keeping these prospects in mind, the future of anomaly detection holds significant promise, and GPT-4V stands at the forefront of this captivating evolution.

![Image](artifacts/image_000002_5a0c2be07945a32fb879500fc664362aa19c0729fbf409719cd302bace4b5649.png)

Figure 3 | Industrial Image Anomaly Detection: Case 1, one-shot, the Bottle category of MVTec AD [6] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000003_64e3ba30850712f7bb38d7aa7707002610c5e0b9ee173c5f9d44d4f5f90c4d05.png)

Figure 4 | Industrial Image Anomaly Detection: Case 2, zero-shot, the Candle category of VisA [116] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000004_7cf4896da3c3cfb0c8302227c8eab80b4ace84022844529f79881685a5e7ce57.png)

Figure 5 | Industrial Image Anomaly Detection: Case 2, one-shot, the Candle category of VisA [116] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000005_851189dac515675cd4730e56c57bfe4699e3cc173b980c72170db1babd829c3d.png)

Figure 6 | Industrial Image Anomaly Detection: Case 3, zero-shot, the PCB2 category of VisA [116] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 7 | Industrial Image Anomaly Detection: Case 3, one-shot, the PCB2 category of VisA [116] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000006_cbc37b28a856914e01dd537c61ad643e4c9cf34ea69cff488484f9507895b938.png)

Figure 8 | Industrial Image Anomaly Localization: Case 1, zero-shot, the Bottle category of MVTec AD [6] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000007_d5cdd2818d48ca8b41341056142f9e7ec5a5a7eabfa5c51ac4e6af5104c76099.png)

Figure 9 | Industrial Image Anomaly Localization: Case 2, the Hazelnut category of MVTec AD [6] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000008_b718d59464906dd861800e33602ba0bdce349c7eae2a0a9bfcdeb7320e701eda.png)

Figure 10 | Industrial Image Anomaly Localization: Case 3, the Capsule category of VisA [116] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000009_fa9fd5b98b537ab34364d29196fed40f05aa49c00ca1bc3ba22c26342b999874.png)

![Image](artifacts/image_000010_04fa06116260a4febd9bd4a1a05a2d763f1ab58518c5f161e86642c93bcb5fe2.png)

Figure 11 | Point Cloud Anomaly Detection: Case 1, zero-shot, the Bagel category of MVTec 3D [8] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000011_eb998929cd53add97293df34e104e725759ddd2b9c854389aabcb8f1996c42a1.png)

Figure 12 | Point Cloud Anomaly Detection: Case 1, one-shot, the Bagel category of MVTec 3D [8] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000012_2ab1c87cc2de679e73c553793c7847b7e62e12024d587c1d0058c83e19d2e499.png)

![Image](artifacts/image_000013_54ee87a00a875713b2fc581bbe51d2698ff76c883da689ee992aa47a1533e84d.png)

Figure 13 | Point Cloud Anomaly Detection: Case 2, zero-shot, the Peach category of MVTec 3D [8] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 14 | Point Cloud Anomaly Detection: Case 2, one-shot, the Peach category of MVTec 3D [8] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000014_203d0f87a9adf82103d035d02d0d06526f55d81cae59cec9e56f2377718d77a2.png)

![Image](artifacts/image_000015_eb998929cd53add97293df34e104e725759ddd2b9c854389aabcb8f1996c42a1.png)

Figure 15 | Point Cloud Anomaly Detection: Case 3, zero-shot, the Rope category of MVTec 3D [8] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 16 | Point Cloud Anomaly Detection: Case 3, one-shot, the Rope category of MVTec 3D [8] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000016_0c8c5df2e043be39e74d0efe7f63ad0398e2736f9214a6ce8dcbfbb71286205e.png)

Figure 17 | Logical Anomaly Detection: Case 1, the Breakfast Box category of MVTec LOCO [7] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000017_2259d1c920e513e7226da7d78d9bc233790686dafb17413f16cc11a71c85e440.png)

![Image](artifacts/image_000018_bc2565f5ceee62488ba7eefa229b4390b3925663730729fd62791d972b169e89.png)

![Image](artifacts/image_000019_76d33971d3882a522c92983e1598f8f2a60c5feb0a04684bc6edf9e85b53e861.png)

Figure 18 | Logical Anomaly Detection: Case 2, the Juice Bottle category of MVTec LOCO [7] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000020_08133afe4c261524fff8acda876f338d8064d9ec498b935145ecc7c603d8ef54.png)

Figure 19 | Logical Anomaly Detection: Case 3, the Splicing Connector category of MVTec LOCO [7] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000021_f0948fd7e055f0d5eb8592d58e9e76e3bacbf4968baba9eeb801bcfd9ed5250b.png)

![Image](artifacts/image_000022_140c45becf9a4aed04f4cfad8cf4e01d0a41d1e3cf236cda16b7fe3705ac6f33.png)

![Image](artifacts/image_000023_98d70f03bcc76bbbdfcc99b49ee2aacfa1c68b34a34ba18b547954a12662ee7f.png)

![Image](artifacts/image_000024_49ed1040381d90cbf78652830d08345263abcc90a4ccbd3dd8b28a34faedb242.png)

Figure 20 | Logical Anomaly Detection: Case 4, the Screw Bag category of MVTec LOCO [7] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 21 | Medical Anomaly Detection: Case 1, the Chest X-ray [49] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000025_ef2c9719d03d0456552e0db16abd318dcb5ef28dcd0ea35719a60f1832cb9c40.png)

![Image](artifacts/image_000026_e25e1e2bfc1d5615ef2418fde9f50fd707ef6b4a5476121cf98fdf40bd607d1a.png)

Figure 22 | Medical Anomaly Detection: Case 1, the Chest X-ray [49] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 23 | Medical Anomaly Detection: Case 2, the Retinal OCT [49] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000027_5582bd5479bde26ebca2de455f59efea169fded546ebe6dccb7da95ad59277e0.png)

Figure 24 | Medical Anomaly Detection: Case 2, the Retinal OCT [49] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000028_33eecfcde62b958fbbad30bb6a5ed6cffc0173dd93026999a39d5702d8172b7b.png)

![Image](artifacts/image_000029_88b6e9c3e882ffd3ccc1a7aa22be29b3017a755ba57f020ef0fad9535f9e9630.png)

Figure 25 | Medical Anomaly Detection: Case 3, the Head CT [51] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 26 | Medical Anomaly Detection: Case 3, the Head CT [51] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000030_46754b3166bd4c807f55f4e5ec53afed683813546f2d07c1000d356021496be4.png)

![Image](artifacts/image_000031_7375517ae58698d6d0cc7f1a8be05b55866a57231df4f96452985c379c07886a.png)

Figure 27 | Medical Anomaly Detection: Case 4, Head MRI Image [18] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000032_6537153dd7ecd535b16e9c899879c2621db09b2a67ecfac467db8a1cd822c71e.png)

![Image](artifacts/image_000033_acb41121cb9ca1f0af5137eec366b4e508df52fcffe953990563c88fdc9f1294.png)

Figure 28 | Medical Anomaly Detection: Case 4, Head MRI Image [18] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

Figure 29 | Medical Anomaly Localization: Case 1, Abdonimal CT Localization [114] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000034_7e4290617512797f8d383ea250221824ad8360c3d42771badd163cea180a1a82.png)

![Image](artifacts/image_000035_e78602f093b0c35cbb4c1452badaa1057f20e0d77e9d6b683deac2a599cdff07.png)

Figure 30 | Medical Anomaly Localization: Case 2,Head MRI Localization [114] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000036_1ff2eee2759908d3c8cc7eb12cf7b1b18f883e94b9784cc27160712f3d3d8223.png)

Figure 31 | Medical Anomaly Localization: Case 3, Skin Lesion Localization [26] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000037_686da4ed9f8fd5dcd0875556ae3552636d63db4f4d4c8dc796da7584f99e86e1.png)

![Image](artifacts/image_000038_b5628aa53124ec718356edca6e9ac120bceba74d4f35480516f44d6c88e342ac.png)

Figure 32 | Medical Anomaly Localization: Case 4, Endoscopy Localization [11] . Yellow highlights the given class information and normal and abnormal state descriptions. Green , red, and blue highlight the expected, incorrect, and additional information outputted by GPT-4V.

![Image](artifacts/image_000039_9717c87cb4010fe2c023834fd15fda20273b5f791392e346ca8c735fdb6b710b.png)

![Image](artifacts/image_000040_9d5dc225a1ce86846df19708291558e767a5e6826056300e3d1d52b7784d4067.png)

![Image](artifacts/image_000041_3ee3c4374b3b0f8db6613171d6c40cd823780c6dadb08fb4e9534b4e92d49833.png)

![Image](artifacts/image_000042_c2a476683c4fd20376ca62c8b3f1ae8c2c9ad27e7479cdbf57f471a5dddfe41f.png)

Figure 33 | Pedestrian Anomaly Detection: Case 1, from UCF-Crime Dataset [85] . Green highlights the expected information outputted by GPT-4V.

Figure 34 | Traffic Anomaly Detection: Case 1, from Kaggle Accident Detection [48] . Green highlights the expected information outputted by GPT-4V.

![Image](artifacts/image_000043_1a6f8643cca0b1941c8a00d51fc6f4d920a1bc40bee552d77b948fdd1f48018f.png)

Figure 35 | Traffic Anomaly Detection: Case 2, from Kaggle Accident Detection [48] . Green highlights the expected information outputted by GPT-4V.

![Image](artifacts/image_000044_a0c21d125c9c20470b3cca9b116a88fedde2a3d672aa53eb357aeb368ad44c6c.png)

Figure 36 | Time Series Anomaly Detection: Case 1, from Outlier Detection Dataset [89].Green highlights the expected information outputted by GPT-4V.

![Image](artifacts/image_000045_f7737872185a596ece54641b67a00590b5c307c3a51d828027820f1d65c9b7da.png)

Figure 37 | Time Series Anomaly Detection: Case 2, from Catfish Sales Dataset [2] . Green highlights the expected information outputted by GPT-4V.

![Image](artifacts/image_000046_edd11d33b8b3ba986341b57afd23a0836310f568d5e300f7e6374807b29fc29a.png)

## References

- [1] Amit Adam, Ehud Rivlin, Ilan Shimshoni, and Daviv Reinitz. Robust real-time unusual event detection using multiple fixed-location monitors. IEEE Transactions on Pattern Analysis and Machine Intelligence , 30(3):555–560, 2008.
- [2] Neptune AI. Anomaly detection in time series. https://neptune.ai/blog/anomaly-detection-in-time-series, 2023. Accessed: 2023-11-04.
- [3] Samet Akçay, Amir Atapour-Abarghouei, and T. Breckon. Ganomaly: Semi-supervised anomaly detection via adversarial training. In Asian Conference on Computer Vision, 2018.
- [4] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [5] Kilian Batzner, Lars Heckler, and Rebecca König. Efficientad: Accurate visual anomaly detection at millisecond-level latencies. arXiv preprint arXiv:2303.14535, 2023.
- [6] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. The MVTec anomaly detection dataset: A comprehensive real-world dataset for unsupervised anomaly detection. International Journal of Computer Vision, 129(4):1038–1059, 2021.
- [7] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization. International Journal of Computer Vision, 130(4):947–969, 2022.
- [8] Paul Bergmann, Xin Jin, David Sattlegger, and Carsten Steger. The MVTec 3d-AD dataset for unsupervised 3d anomaly detection and localization. In Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, pages 202–213, 2022.
- [9] Paul Bergmann and David Sattlegger. Anomaly detection in 3d point clouds using deep geometric descriptors. arXiv preprint arXiv:2202.11660, 2022.
- [10] Ane Bl'azquez-Garc'ia, Angel Conde, Usue Mori, and José Antonio Lozano. A review on outlier/anomaly detection in time series data. ACM Computing Surveys, 54:1 – 33, 2020.
- [11] Hanna Borgli, Vajira Thambawita, Pia H Smedsrud, Steven Hicks, Debesh Jha, Sigrun L Eskeland, Kristin Ranheim Randel, Konstantin Pogorelov, Mathias Lux, Duc Tien Dang Nguyen, et al. Hyperkvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. Scientific data , 7(1):283, 2020.
- [12] Yuxuan Cai, Dingkang Liang, Dongliang Luo, Xinwei He, Xin Yang, and Xiang Bai. A discrepancy aware framework for robust anomaly detection. IEEE Transactions on Industrial Informatics, pages 1–10, 2023.
- [13] Yunkang Cao, Yanan Song, Xiaohao Xu, Shuya Li, Yuhao Yu, Yifeng Zhang, and Weiming Shen. Semi-supervised knowledge distillation for tiny defect detection. In 2022 IEEE 25th International Conference on Computer Supported Cooperative Work in Design (CSCWD), pages 1010–1015. IEEE, 2022.
- [14] Yunkang Cao, Qian Wan, Weiming Shen, and Liang Gao. Informative knowledge distillation for image anomaly segmentation. Knowledge-Based Systems, 248:108846, 2022.
- [15] Yunkang Cao, Xiaohao Xu, Zhaoge Liu, and Weiming Shen. Collaborative discrepancy optimization for reliable image anomaly localization. IEEE Transactions on Industrial Informatics, pages 1–10, 2023.
- [16] Yunkang Cao, Xiaohao Xu, and Weiming Shen. Complementary pseudo multimodal feature for point cloud anomaly detection. arXiv preprint arXiv:2303.13194, 2023.
- [17] Yunkang Cao, Xiaohao Xu, Chen Sun, Yuqi Cheng, Zongwei Du, Liang Gao, and Weiming Shen. Segment any anomaly without training via hybrid prompt regularization. arXiv preprint arXiv:2305.10724, 2023.
- [18] Navoneel Chakrabarty. Brain mri images for brain tumor detection, 2019.
- [19] Raghavendra Chalapathy and Sanjay Chawla. Deep learning for anomaly detection: A survey. arXiv preprint arXiv:1901.03407, 2019.

- [20] Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. ACM Computing Surveys, 41(3), jul 2009.
- [21] Rui Chen, Guoyang Xie, Jiaqi Liu, Jinbao Wang, Ziqi Luo, Jinfan Wang, and Feng Zheng. Easynet: An easy network for 3d industrial anomaly detection. Proceedings of the 31st ACM International Conference on Multimedia, 2023.
- [22] Xuhai Chen, Yue Han, and Jiangning Zhang. A zero-/few-shot anomaly classification and segmentation method for CVPR 2023 VAND workshop challenge tracks 1&amp;2: 1st place on zero-shot AD and 4th place on few-shot AD. arXiv preprint arXiv:2305.17382, 2023.
- [23] Xiaoran Chen, Suhang You, Kerem Can Tezcan, and Ender Konukoglu. Unsupervised lesion detection via image restoration with a normative prior. Medical image analysis, 64:101713, 2020.
- [24] Yong Shean Chong and Yong Haur Tay. Abnormal event detection in videos using spatiotemporal autoencoder. In International symposium on neural networks, pages 189–196. Springer, 2017.
- [25] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [26] Noel CF Codella, David Gutman, M Emre Celebi, Brian Helba, Michael A Marchetti, Stephen W Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, et al. Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). In 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018), pages 168–172. IEEE, 2018.
- [27] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
- [28] Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. Padim: a patch distribution modeling framework for anomaly detection and localization. In International Conference on Pattern Recognition, pages 475–489. Springer, 2021.
- [29] Jan Diers and Christian Pigorsch. A survey of methods for automated quality control based on images. International Journal of Computer Vision, 2023.
- [30] Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar. Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017.
- [31] Tharindu Fernando, Harshala Gammulle, Simon Denman, Sridha Sridharan, and Clinton Fookes. Deep learning for medical anomaly detection–a survey. ACM Computing Surveys (CSUR), 54(7):1–37, 2021.
- [32] Alberto Floris, Luca Frittoli, Diego Carrera, and Giacomo Boracchi. Composite layers for deep anomaly detection on 3d point clouds. arXiv preprint arXiv:2209.11796, 2022.
- [33] Harrou Fouzi and Ying Sun. Enhanced anomaly detection via pls regression models and information entropy theory. In IEEE Symposium Series on Computational Intelligence (SSCI), pages 383–388, 2015.
- [34] Jingkun Gao, Xiaomin Song, Qingsong Wen, Pichao Wang, Liang Sun, and Huan Xu. Robusttad: Robust time series anomaly detection via decomposition and convolutional neural networks. arXiv preprint arXiv:2002.09545, 2020.
- [35] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1705–1714, 2019.
- [36] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans, 2023.
- [37] Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, and Jinqiao Wang. Anomalygpt: Detecting industrial anomalies using large vision-language models. arXiv preprint arXiv:2308.15366 , 2023.

- [38] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning temporal regularity in video sequences. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 733–742, 2016.
- [39] Lars Heckler, Rebecca König, and Paul Bergmann. Exploring the importance of pretrained feature extractors for unsupervised anomaly detection and localization. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2917–2926, 2023.
- [40] Chaoqin Huang, Haoyan Guan, Aofan Jiang, Ya Zhang, Michael Spratling, and Yan-Feng Wang. Registration based few-shot anomaly detection. In European Conference on Computer Vision, pages 303–319. Springer, 2022.
- [41] Chao Huang, Chengliang Liu, Jie Wen, Lian Wu, Yong Xu, Qiuping Jiang, and Yaowei Wang. Weakly supervised video anomaly detection via self-guided temporal discriminative transformer. IEEE Transactions on Cybernetics, pages 1–14, 2022.
- [42] Chao Huang, Jie Wen, Yong Xu, Qiuping Jiang, Jian Yang, Yaowei Wang, and David Zhang. Selfsupervised attentive generative adversarial networks for video anomaly detection. IEEE Transactions on Neural Networks and Learning Systems, pages 1–15, 2022.
- [43] Chao Huang, Zehua Yang, Jie Wen, Yong Xu, Qiuping Jiang, Jian Yang, and Yaowei Wang. Selfsupervision-augmented deep autoencoder for unsupervised visual anomaly detection. IEEE Transactions on Cybernetics, 52(12):13834–13847, 2022-12.
- [44] Tsuyoshi Idé, Ankush Khandelwal, and Jayant Kalagnanam. Sparse gaussian markov random field mixtures for anomaly detection. In IEEE 16th International Conference on Data Mining (ICDM), pages 955–960, 2016.
- [45] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. Winclip: Zero-/few-shot anomaly classification and segmentation. arXiv preprint arXiv:2303.14814 , 2023.
- [46] Yuxin Jiang, Yunkang Cao, and Weiming Shen. A masked reverse knowledge distillation method incorporating global and local information for image anomaly detection. Knowledge-Based Systems , 280:110982, 2023.
- [47] Yang Jiao, Kai Yang, Dongjing Song, and Dacheng Tao. Timeautoad: Autonomous anomaly detection with self-supervised contrastive loss for multivariate time series. IEEE Transactions on Network Science and Engineering, 9(3):1604–1619, 2022.
- [48] C. Kay. Accident detection from cctv footage. https://www.kaggle.com/datasets/ckay16/accident-detection-fromcctv-footage, 2022. Kaggle dataset.
- [49] Daniel S Kermany, Michael Goldbaum, Wenjia Cai, Carolina CS Valentim, Huiying Liang, Sally L Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. cell, 172(5):1122–1131, 2018.
- [50] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , pages 4015–4026, October 2023.
- [51] Felipe Campos Kitamura. Head ct - hemorrhage, 2018.
- [52] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020, 2023.
- [53] Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister. Cutpaste: Self-supervised learning for anomaly detection and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9664–9674, 2021.
- [54] Yufei Liang, Jiangning Zhang, Shiwei Zhao, Runze Wu, Yong Liu, and Shuwen Pan. Omni-frequency channel-selection representations for unsupervised anomaly detection. arXiv preprint arXiv:2203.00259 , 2022.

- [55] Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v (ision), llava-1.5, and other multi-modality models. arXiv preprint arXiv:2310.14566, 2023.
- [56] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023.
- [57] Fuxiao Liu, Yaser Yacoob, and Abhinav Shrivastava. Covid-vts: Fact extraction and verification on short video platforms. arXiv preprint arXiv:2302.07919, 2023.
- [58] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [59] Jiaqi Liu, Guoyang Xie, Rui Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, and Feng Zheng. Real3d-ad: A dataset of point cloud anomaly detection. arXiv preprint arXiv:2309.13226, 2023.
- [60] Mingxuan Liu, Yunrui Jiao, and Hong Chen. Skip-st: Anomaly detection for medical images using student-teacher network with skip connections. In 2023 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–5, 2023.
- [61] Mingxuan Liu, Yunrui Jiao, Hongyu Gu, Jingqiao Lu, and Hong Chen. Data augmentation using image-to-image translation for tongue coating thickness classification with imbalanced data. In 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), pages 90–94, 2022.
- [62] Mingxuan Liu, Yunrui Jiao, Jingqiao Lu, and Hong Chen. Anomaly detection for medical images using teacher-student model with skip connections and multi-scale anomaly consistency. TechRxiv, 2023.
- [63] Tongkun Liu, Bing Li, Xiao Du, Bingke Jiang, Xiao Jin, Liuyi Jin, and Zhu Zhao. Component-aware anomaly detection framework for adjustable and logical industrial visual inspection. arXiv preprint arXiv:2305.08509, 2023.
- [64] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In IEEE International Conference on Computer Vision, pages 2720–2727, 2013.
- [65] Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, and Ruimin Hu. Hierarchical vector quantized transformer for multi-class unsupervised anomaly detection. arXiv preprint arXiv:2310.14228, 2023.
- [66] Weixin Luo, Wen Liu, and Shenghua Gao. Remembering history with convolutional lstm for anomaly detection. In IEEE International Conference on Multimedia and Expo (ICME), pages 439–444, 2017.
- [67] Weixin Luo, Wen Liu, and Shenghua Gao. Remembering history with convolutional lstm for anomaly detection. In 2017 IEEE International conference on multimedia and expo (ICME), pages 439–444. IEEE, 2017.
- [68] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In Proceedings of the IEEE International Conference on Computer Vision , pages 341–349, 2017.
- [69] Vijay Mahadevan, Weixin Li, Viral Bhalodia, and Nuno Vasconcelos. Anomaly detection in crowded scenes. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 1975–1981, 2010.
- [70] Jefferson Ryan Medel and Andreas Savakis. Anomaly detection in video using predictive convolutional long short-term memory networks. arXiv preprint arXiv:1612.00390, 2016.
- [71] OpenAI. Gpt-4v(ision) system card. 2023.
- [72] Guansong Pang, Chunhua Shen, Longbing Cao, and Anton van den Hengel. Deep learning for anomaly detection. ACM Computing Surveys, 54:1 – 38, 2020.
- [73] Hyunjong Park, Jongyoun Noh, and Bumsub Ham. Learning memory-guided normality for anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14372–14381, 2020.
- [74] Mahdyar Ravanbakhsh, Moin Nabi, Enver Sangineto, Lucio Marcenaro, Carlo Regazzoni, and Nicu Sebe. Abnormal event detection in videos using generative adversarial nets. In IEEE International Conference on Image Processing (ICIP), pages 1577–1581, 2017.

- [75] Tal Reiss, Niv Cohen, Liron Bergman, and Yedid Hoshen. Panda: Adapting pretrained features for anomaly detection and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2806–2814, 2021.
- [76] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14318–14328, 2022.
- [77] Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt. Asymmetric student-teacher networks for industrial anomaly detection. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2591–2601, 2022.
- [78] Lukas Ruff, Jacob R. Kauffmann, Robert A. Vandermeulen, Gregoire Montavon, Wojciech Samek, Marius Kloft, Thomas G. Dietterich, and Klaus-Robert Muller. A unifying review of deep and shallow anomaly detection. Proceedings of the IEEE, 109(5):756–795, 2021.
- [79] Mayu Sakurada and Takehisa Yairi. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, MLSDA'14, page 4–11, New York, NY, USA, 2014. Association for Computing Machinery.
- [80] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.
- [81] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11987–11997, October 2023.
- [82] Kihyuk Sohn, Chun-Liang Li, Jinsung Yoon, Minho Jin, and Tomas Pfister. Learning and evaluating representations for deep one-class classification. In International Conference on Learning Representations , 2020.
- [83] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery &amp; data mining, pages 2828–2837, 2019.
- [84] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.
- [85] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6479–6488, 2018.
- [86] Hanlin Tan, Yongping Zhai, Yu Liu, and Maojun Zhang. Fast anomaly detection in traffic surveillance video based on robust sparse optical flow. In IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 1976–1980, 2016.
- [87] Yu Tian, Fengbei Liu, Guansong Pang, Yuanhong Chen, Yuyuan Liu, Johan W Verjans, Rajvinder Singh, and Gustavo Carneiro. Self-supervised pseudo multi-class pre-training for unsupervised anomaly detection and segmentation in medical images. Medical Image Analysis, 90:102930, 2023.
- [88] Yu Tian, Guansong Pang, Fengbei Liu, Yuanhong Chen, Seon Ho Shin, Johan W Verjans, Rajvinder Singh, and Gustavo Carneiro. Constrained contrastive distribution learning for unsupervised anomaly detection and localisation in medical images. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part V 24, pages 128–140. Springer, 2021.
- [89] Stack Exchange User. Simple outlier detection for time series. https://stats.stackexchange.com/questions/ 427327/simple-outlier-detection-for-time-series, 2021. Accessed: 2023-11-04.
- [90] Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, and Abhijit Mahalanobis. Attention guided anomaly localization in images. In European Conference on Computer Vision, pages 485–503. Springer, 2020.
- [91] Qian Wan, Yunkang Cao, Liang Gao, Weiming Shen, and Xinyu Li. Position encoding enhanced feature mapping for image anomaly detection. In 2022 IEEE 18th International Conference on Automation Science and Engineering (CASE), pages 876–881. IEEE, 2022-08-20.

- [92] Qian Wan, Liang Gao, and Xinyu Li. Logit inducing with abnormality capturing for semi-supervised image anomaly detection. IEEE Transactions on Instrumentation and Measurement, 71:1–12, 2022.
- [93] Qian Wan, Liang Gao, Xinyu Li, and Long Wen. Industrial image anomaly localization based on gaussian clustering of pretrained feature. IEEE Transactions on Industrial Electronics, 69(6):6182–6192.
- [94] Qian Wan, Liang Gao, Xinyu Li, and Long Wen. Unsupervised image anomaly detection and segmentation based on pretrained feature mapping. 19(3):2330–2339, 2023-03.
- [95] Guodong Wang, Shumin Han, Errui Ding, and Di Huang. Student-teacher feature pyramid matching for anomaly detection. In British Machine Vision Conference, 2021.
- [96] Yue Wang, Jinlong Peng, Jiangning Zhang, Ran Yi, Yabiao Wang, and Chengjie Wang. Multimodal industrial anomaly detection via hybrid fusion. arXiv preprint arXiv:2303.00601, 2023.
- [97] Chaoyi Wu, Jiayu Lei, Qiaoyu Zheng, Weike Zhao, Weixiong Lin, Xiaoman Zhang, Xiao Zhou, Ziheng Zhao, Ya Zhang, Yanfeng Wang, and Weidi Xie. Can GPT-4v(ision) serve medical applications? case studies on GPT-4v for multimodal medical diagnosis. arXiv preprint arXiv:2310.09909, 2023.
- [98] Guoyang Xie, Jinbao Wang, Jiaqi Liu, Jiayi Lyu, Yong Liu, Chengjie Wang, Feng Zheng, and Yaochu Jin. IM-IAD: Industrial image anomaly detection benchmark in manufacturing. arXiv preprint arXiv:2301.13359, 2023.
- [99] Guoyang Xie, Jingbao Wang, Jiaqi Liu, Feng Zheng, and Yaochu Jin. Pushing the limits of fewshot anomaly detection in industry vision: Graphcore. In International Conference on Learning Representations, 2023.
- [100] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in GPT-4v. arXiv preprint, arXiv:2310.11441, 2023.
- [101] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421 , 2023.
- [102] Haiming Yao, Wei Luo, Wenyong Yu, Xiaotian Zhang, Zhenfeng Qiang, Donghao Luo, and Hui Shi. Dual-attention transformer and discriminative flow for industrial visual anomaly detection. IEEE Transactions on Automation Science and Engineering, pages 1–15, 2023.
- [103] Haiming Yao, Wenyong Yu, Wei Luo, Zhenfeng Qiang, Donghao Luo, and Xiaotian Zhang. Learning global-local correspondence with semantic bottleneck for logical anomaly detection. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2023.
- [104] Mingze Yuan, Yingda Xia, Hexin Dong, Zifan Chen, Jiawen Yao, Mingyan Qiu, Ke Yan, Xiaoli Yin, Yu Shi, Xin Chen, et al. Devil is in the queries: Advancing mask transformers for real-world medical image segmentation and out-of-distribution localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23879–23889, 2023.
- [105] Andrei Zaharescu and Richard Wildes. Anomalous behaviour detection using spatiotemporal oriented energies, subset inclusion histogram comparison and event-driven processing. In European Conference on Computer Vision, pages 563–576. Springer, 2010.
- [106] J. Zhang, Masanori Suganuma, and Takayuki Okatani. Contextual affinity distillation for image anomaly detection. arXiv preprint arXiv:2307.03101,, 2023.
- [107] Jianpeng Zhang, Yutong Xie, Yi Li, Chunhua Shen, and Yong Xia. Covid-19 screening on chest x-ray images using deep learning based anomaly detection. arXiv preprint arXiv:2003.12338, 2020.
- [108] Yuxin Zhang, Jindong Wang, Yiqiang Chen, Han Yu, and Tao Qin. Adaptive memory networks with self-supervised learning for unsupervised anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 2022.
- [109] Bin Zhao, Li Fei-Fei, and Eric P. Xing. Online detection of unusual events in videos via dynamic sparse coding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3313–3320, 2011.
- [110] Hang Zhao, Yujing Wang, Juanyong Duan, Congrui Huang, Defu Cao, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. Multivariate time-series anomaly detection via graph attention network. In 2020 IEEE International Conference on Data Mining (ICDM), pages 841–850. IEEE, 2020.

- [111] Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, and Hao Zhao. Pad: A dataset and benchmark for pose-agnostic anomaly detection. arXiv preprint arXiv:2310.07716 , 2023.
- [112] Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, and Jiming Chen. Anomalyclip: Object-agnostic prompt learning for zero-shot anomaly detection. arXiv preprint arXiv:2310.18961, 2023.
- [113] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592 , 2023.
- [114] David Zimmerer, Peter M. Full, Fabian Isensee, Paul Jäger, Tim Adler, Jens Petersen, Gregor Köhler, Tobias Ross, Annika Reinke, Antanas Kascenas, Bjørn Sand Jensen, Alison Q. O'Neil, Jeremy Tan, Benjamin Hou, James Batten, Huaqi Qiu, Bernhard Kainz, Nina Shvetsova, Irina Fedulova, Dmitry V. Dylov, Baolun Yu, Jianyang Zhai, Jingtao Hu, Runxuan Si, Sihang Zhou, Siqi Wang, Xinyang Li, Xuerun Chen, Yang Zhao, Sergio Naval Marimont, Giacomo Tarroni, Victor Saase, Lena Maier-Hein, and Klaus Maier-Hein. Mood 2020: A public benchmark for out-of-distribution detection and localization on medical images. IEEE Transactions on Medical Imaging, 41(10):2728–2738, 2022.
- [115] Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, and Haifeng Chen. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In International conference on learning representations, 2018.
- [116] Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. Spot-the-difference selfsupervised pre-training for anomaly detection and segmentation. In European Conference on Computer Vision, pages 392–408. Springer, 2022.