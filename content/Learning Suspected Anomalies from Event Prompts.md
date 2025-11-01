## Learning Suspected Anomalies from Event Prompts for Video Anomaly Detection

Chenchen Tao ∗ , Xiaohao Peng ∗ , Chong Wang R , Member, IEEE, Jiafei Wu, Puning Zhao, Jun Wang, Jiangbo Qian

Abstract—Most models for weakly supervised video anomaly detection (WS-VAD) rely on multiple instance learning, aiming to distinguish normal and abnormal snippets without specifying the type of anomaly. However, the ambiguous nature of anomaly definitions across contexts may introduce inaccuracy in discriminating abnormal and normal events. To show the model what is anomalous, a novel framework is proposed to guide the learning of suspected anomalies from event prompts. Given a textual prompt dictionary of potential anomaly events and the captions generated from anomaly videos, the semantic anomaly similarity between them could be calculated to identify the suspected events for each video snippet. It enables a new multi-prompt learning process to constrain the visual-semantic features across all videos, as well as provides a new way to label pseudo anomalies for self-training. To demonstrate its effectiveness, comprehensive experiments and detailed ablation studies are conducted on four datasets, namely XD-Violence, UCF-Crime, TAD, and ShanghaiTech. Our proposed model outperforms most state-of-the-art methods in terms of AP or AUC (86.5%, 90.4%, 94.4%, and 97.4%). Furthermore, it shows promising performance in openset and cross-dataset cases. The data, code, and models can be found at: https://github.com/shiwoaz/lap.

Index Terms—Weakly Supervised, Video Anomaly Detection, Event Prompt, Multiple Instanse Learning.

## I. INTRODUCTION

V IDEO anomaly detection (VAD) [1], [2], [3] is crucial in video surveillance, given the extensive use of surveillance cameras. The task of VAD is to determine whether each frame in a video is normal or abnormal, which poses a significant challenge as it is not feasible to train a model with complete supervision. Consequently, weakly supervised learning methods (WS-VAD) [4], [5], [6] that solely rely on video-level annotations have gained importance and popularity in recent years.

The general paradigm of these methods involves utilizing convolutional networks such as 3D ConvNet (C3D) [7], inflated 3D ConvNet (I3D) [8], or vision transformer [9] to extract visual features and aggregate spatio-temporal information between consecutive frames. Subsequently, an anomalydetection network is trained using multiple instance learning (MIL) [10]. This approach simultaneously maximizes and minimizes the top-k highest scores from individual anomaly and normal videos, respectively. Most methods [10], [11] only focused on the visual-related modality, while some [12], [13] have incorporated semantic descriptions into videos. However,

∗ These authors contributed equally to this work and should be considered co-first authors.

R Corresponding Author: Chong Wang.

Fig. 1. The difference between the traditional multiple instance learning methods (upper) and our model (lower). The former one only learns the anomalies using top-k scores in each abnormal video, while the latter utilizes a prompt dictionary to provide extra guidance across different videos.

![Image](artifacts/image_000000_f0b78b61f2d1a554478a7aaa07669acb8581343eb089e613837715a1f37b89fc.png)

such semantic information was simply fused with the visual one, instead of delving into the underlying meaning of the textual descriptions. As a result, the MIL based approaches often suffer from a relatively high false alarm rate (FAR) and low accuracy in detecting ambiguous abnormal events.

Meanwhile, foundation models in natural language processing (NLP) and computer vision (CV), such as InstructGPT [14] and CLIP [15], have demonstrated impressive performance on multimodal tasks. Additionally, prompting techniques in the image field provide a new way to transfer semantic information from well-trained foundation models into vision tasks. It is intriguing to explore whether CLIP's zero-shot detection ability can be effectively transferred into video anomaly detection.

Therefore, a novel framework to Learn suspected Anomalies from event Prompts, called LAP, is proposed in this paper. As illustrated in Figure 1, a prompt dictionary is designed to list the potential anomaly events. In order to mark suspected anomalies, it is utilized to compare with the captions generated from anomaly videos in the form of semantic features. As a result, an anomaly vector that records the most suspected anomalous events for each video snippet can be obtained. This vector is used to guide a new multi-prompt learning scheme across different videos, as well as form a new set of pseudo anomaly labels.

The main contributions of this work are threefold:

- The new textual prompts describing the abnormal events are introduced into weakly supervised video anomaly detection. Giving the explanation of what is anomalous, the score predictor can implicitly learn more details about

- the anomalies. It leads to incredible performance on openset and cross-database problems.
- A new multi-prompt learning strategy is proposed to provide an overall understanding of normal and abnormal patterns across different videos, while MIL is limited to individual videos.
- Additional pseudo labels are excavated from the anomaly videos according to the semantic similarity between the event prompts and videos. They are utilized to train the predictor effectively in a self-supervised manner.

## II. RELATED WORKS

## A. Weakly Supervised Video Anomaly Detection

The weakly supervised methods tackle frame-level anomaly detection by video-level annotations. Most of them are based primarily on multiple instance learning (MIL) due to limited annotated labels [10]. However, conventional MIL faces challenges in providing sufficient supervision for various anomalies, leading to misclassifications and a high false alarm rate. To address these issues, Yu et al. propose cross-epoch learning (XEL) [4], which stores hard instances from previous epochs to optimize the anomaly predictor in the latest epoch. Additionally, dual memory units with uncertainty regulation (URDMU) [16] extend the anomaly memory unit into learnable dual memory units to alleviate the high false alarm rate issue. Another approach called robust temporal feature magnitude learning (RTFM) [11] trains a feature magnitude learning function to effectively recognize positive instances. All these methods are based on single or multiple visual modalities, including RGB and optical flow.

As the field embraces multi-modality models like GPT [14] and CLIP [15], researchers are now focusing on text-visual models. Text-empowered video anomaly detection (TEVAD) [12] demonstrates improvements by generating text and visual features independently. However, TEVAD treats text features as auxiliary to visual features. In contrast, our approach aims to capitalize on the high semantic-level guidance provided by text, offering a unique perspective for enhancing anomaly detection performance.

## B. Prompt Tuning for Visual Tasks

In the realm of pre-trained foundation multimodality models, a cost-effective prompt tuning approach is gaining traction for adapting models to downstream tasks in the domains of natural language processing (NLP) [17], [18] and computer vision [19].

The concept of prompt tuning originated in computer vision to tackle zero-shot or few-shot image tasks by incorporating semantic guidance. Multimodal models such as CLIP [15] leverage textual prompts for image classification, demonstrating state-of-the-art performance. In the video domain, Sato [20] explores prompt tuning for zero-shot anomaly action recognition, using skeleton features and text embeddings in a shared space to refine decision boundaries. Wang et al. introduce prompt learning for action recognition (PLAR) [21], which incorporates optical flow and learnable prompts to acquire input-invariant knowledge from a prompt expert dictionary and input-specific knowledge based on the data.

A previous effort, the prompt-based feature mapping framework (PFMF) [22], applies prompt-based learning to semisupervised video anomaly detection. PFMF generates anomaly prompts by concatenating anomaly vectors from virtual datasets and scene vectors from real datasets, guiding the feature mapping network. However, the prompt in PFMF defines anomalies at the visual level, introducing ambiguity. In our work, we propose textual anomaly prompts based on prior knowledge to mine fine-grained anomalies to achieve high performance.

Several recent studies have introduced additional information or tasks to maximize the CLIP's effectiveness in WSVAD. CLIP-assisted temporal self-attention (CLIP-TSA) [23] incorporates temporal information into CLIP features using a self-attention mechanism. In contrast, VadCLIP [13] delves deeper into aligning textual category labels with CLIP's visual features to enhance its WS-VAD performance. Unlike VadCLIP, which constructs learnable prompts based on class labels, our approach designs event prompts to describe specific anomaly-related situations, eliminating the need for additional supervised information or learning tasks.

## III. METHODOLOGY

The proposed LAP framework, as shown in Figure 2, is built upon the basic VAD structure consisting of a visual feature extractor and a score predictor. To enhance discrimination between normal and abnormal videos, semantic clues from anomaly events are integrated using a prompt dictionary and an additional semantic feature extractor. This integration introduces three key processes: feature synthesis, multi-prompt learning, and pseudo anomaly labeling. Semantic features are extracted from videos and fused with visual features, enriching the overall representation. Simultaneously, anomaly prompts, describing abnormal events, are employed to generate another set of semantic features. An anomaly similarity matrix is then computed between these two semantic feature sets. This matrix identifies the most anomalous features corresponding to each prompt in the dictionary. This batch-level anomaly vector not only facilitates a new multi-prompt learning procedure but also acts as a set of snippet-level pseudo labels. The subsequent subsections delve into the specifics of these procedures.

## A. Feature Synthesis

Following the protocol of WS-VAD [24], we adopt a training approach using pairwise normal and abnormal data. Each training batch comprises an abnormal bag and a normal bag, consisting of b abnormal and normal videos with labels y a = 1 ∈ R b×1 and y n = 0 ∈ R b×1 , respectively. In this setup, every video is divided into L snippets, each containing 16 consecutive frames. Consequently, the total number of snippets in each bag is N = b × L. All of these snippets are then processed by the visual and semantic feature extractors.

To clarify, we acquire the visual features V a ∈ R N×d v and V n ∈ R N×d v from video snippets in the abnormal and normal bags, utilizing the visual encoder of a CLIP model [15]. Given

Fig. 2. The overview of the proposed LAP framework. Synthetic features, as input to score predictors, are generated through the visual and semantic feature extractors. A prompt dictionary is used to produce the anomaly matrix and vector, which is employed to perform multi-prompt learning (MPL) and pseudo anomaly labeling (PAL) across different videos.

![Image](artifacts/image_000001_35711b94eb911e1cca2cfc6cb62db70a209da23bf7429162fbe84a791b71e93f.png)

that many VAD videos, primarily from surveillance, often lack associated text descriptions, we leverage a pre-trained visualto-text encoder from SwinBERT [25], following TEVAD [12], to generate descriptions for each video snippet. These textual descriptions then undergo processing by the semantic feature extractor (SimCSE [26]), producing corresponding semantic features T a ∈ R N×d t and T n ∈ R N×d t for abnormal and normal video snippets. With extracted visual features and semantic features in hand, we feed these features into a multiscale temporal network (MTN) to obtain both local and global temporal fused features.

Intuitively, a combination of visual and semantic features is employed to synthesize new features F a ∈ R N×df and F n ∈ R N×df , aiming for an enhanced feature representation,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where θ symbolizes a feature alignment and fusion operation. It can be either a concatenation or addition. Subsequently, the anomaly scores s a and s n can be calculated by applying F a and F n to a score predictor. Typically, this predictor takes the form of a multi-layer perceptron (MLP) [27], expressed as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B. Multi-Prompt Learning

In recent WS-VAD, the prevalent approach for training the anomaly score predictor involves a multiple instance learning (MIL) framework [12], [28]. This framework selects the top-k highest anomaly scores from each video, whether abnormal or normal and employs their average ˆ ˆy as the predicted value for the respective video. Given the complete score set s = [s a ;sn] ∈ R 2N×1 ,

<!-- formula-not-decoded -->

where maxk(s) denotes the operator to select k largest values from vector s , k is usually from 2 to 5, i and j indicate the snippet and video indices, respectively. Then the MIL loss L MIL is formulated as,

<!-- formula-not-decoded -->

From Equation 5, it can be seen that the top-k strategy focuses only on a few snippets with the highest scores within an individual video. Moreover, the top anomaly scores in an abnormal video may not be from an abnormal snippet. Therefore, a new textual prompt dictionary consisting of P anomaly prompts is designed to link abnormal video snippets from different videos. Unlike the category annotations used in VadCLIP [13], we expanded the single word annotations into complete anomaly sentences, like "someone is doing something to whom" or "something is what". These sentences can better describe the events/actions related to a certain anomaly category. Then, the prompt dictionary is constructed as a set of these anomaly sentences, e.g. "A man is shooting a gun" or "Something is on fire". As depicted in Figure 2, these prompts undergo the same semantic extraction process (SimCSE [26]) as the earlier video captions, generating their respective semantic features M ∈ R P ×d t . Subsequently, we calculate the similarity between each prompt in the dictionary

Fig. 3. Visualization of the proposed anomaly matrix Ψ ⊤ . It is truncated due to the limited column width.

![Image](artifacts/image_000002_93ab115d690a36afe995ddce752a2d89c124809976676596903a479cfb34733d.png)

and every snippet in T a to construct an anomaly matrix Ψ ∈ R N×P as,

<!-- formula-not-decoded -->

in which || · || denotes the l 2 -norm. The consideration of T n is dismissed here since there are no abnormal snippets in the normal bag. In essence, each element in Ψ provides insights into the probable type of anomaly associated with each snippet or indicates where a predetermined abnormal event might occur. Figure 3 offers a visual representation of Ψ, where frames containing abnormal events exhibit more pronounced colors. Notably, there is a discernible alignment between the frames and prompts.

In order to exploit the anomaly features across different videos, the most likely anomalous event of each snippet, i.e. the highest values in each row of Ψ, is picked to construct a new anomaly vector c ∈ R N×1 .

To leverage these potential anomaly samples, we introduce a novel multi-prompt learning strategy. Based on the predicted score s and the anomaly vector c, all features in F n and F a are categorized into three sets: anchor set, positive set, and negative set. Subsequently, their averages are computed, denoted as fa fanc, fp fpos , and fn fneg . It's important to note that fa fanc and fp fpos model the normal features in normal and abnormal videos, respectively, and can be expressed as,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where argmin P (s) denotes the operator to obtain the indices of P lowest values in vector s , F n [i, :] and F a [i, :] are the i-th row of F n and F a , respectively, which is a synthetic feature vector to represent a certain video snippet. In contrast, the negative set is built by choosing the most anomalous samples in anomaly videos, according to the similarity values in c . Thus the feature fn fneg can be formulated as,

<!-- formula-not-decoded -->

where argmax P (c) denotes the operator to obtain the indices of P largest values in vector c .

Based on these representative features, it is possible to provide an overall understanding of normal and abnormal patterns across different videos. Thus, the multi-prompt learning loss L MPL is defined in a form of triplet loss,

<!-- formula-not-decoded -->

where α represents the margin coefficient. The goal of LMPL is to establish a considerable distance between fn fneg and both fa fanc and fp fpos while simultaneously bringing fa fanc and fp fpos closer together. This feature-level examination implicitly impacts the training of the score predictor, given that the selection of fa fanc and fp fpos is based on s .

## C. Pseudo Anomaly Labeling

In addition to constructing the negative set in MPL, the anomaly vector c serves as a metric for pseudo-labels, enabling the extraction of more latent information in the anomaly bag T a . Specifically, the snippet-level pseudo-anomaly label p is determined by a dynamic threshold within the current batch,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p[i] and c[i] are the i-th element of p and c, mean{c} and std{c} are the mean and standard deviation considering the anomaly vector c, and τ is a hyper-parameter. Then, the anomaly score predictor can be trained in a fully supervised manner, through a pseudo anomaly loss LPAL ,

<!-- formula-not-decoded -->

By incorporating prior knowledge into the pseudo label, the PAL module can better distinguish fine-grained anomalies and generate more accurate detecting results across abnormal videos.

TABLE I PERFORMANCE COMPARISON OF STATE -OF -THE -ART METHODS ON XD-VIOLENCE (AP%) AND UCF-CRIME (AUC%). BOLD AND UNDERLINE INDICATE THE BEST AND SECOND -BEST RESULTS .

| Type   | Source    | Method         | Feat.                        | XD    | UCF UC AUC   | UCF UC AUC   |
|--------|-----------|----------------|------------------------------|-------|---------------|---------------|
| Typ    | Source    | Method         | Feat.                        | AP    | AUC  all     | AUC abn      |
| emi    | CVPR 16’  | Conv-AE [29]   | AE                           | 30.7  | -             | -             |
| Sem    | CVPR 22’  | GCL [30]       | CNN                          | -     | 71.0          | -             |
| y      | ICCV 21’  | RTFM [11]      | CLIP                         | 78.3  | 85.7          | 63.9          |
|        | AAAI 22’  | MSL [31]       | ViT                          | 78.6  | 85.6          | -             |
|        | ECCV 22’  | CSL-TAL [32]   | I3D                          | 71.7  | -             | -             |
|        | CVPR 22’  | BN-SVP [33]    | I3D                          | -     | 83.4          | -             |
|        | CSVT 23’  | Yang [34]      | I3D                          | 77.7  | 81.5          | -             |
| kly    | AAAI 23’  | UR-DMU [16]    | CLIP                         | 82.4  | 86.7          | 68.6          |
| Weak   | CVPR 23’  | ECUPL [35]     | I3D                          | 81.4  | 86.2          | -             |
| We     | CVPR 23’  | CMRL [36]      | I3D                          | 81.3  | 86.1          | -             |
|        | CVPR 23’  | TEVAD [12]     | I3D                          | 79.8  | 84.9          | -             |
|        | AAAI 23’  | MGFN [37]      | ViT                          | 80.1  | -             | -             |
|        | CVPR 23’  | UMIL [28]      | XCLIP                        | -     | 86.7          | 68.7          |
|        | ICIP 23’  | CLIP-TSA [23]  | CLIP                         | 82.2  | 87.6          | -             |
|        | CVPR 24’  | Wu et al. [38] | CLIP                         | 66.5  | 86.4          | -             |
|        | AAAI 24’  | VadCLIP [13]   | CLIP                         | 84.5  | 88.0          | 70.2          |
|        | ours      | LAP            | CLIP (SwinBert) CLIP (CA) | 86.5  | 88.9          | 73.0          |

The final training loss LLAP can be denoted as,

<!-- formula-not-decoded -->

where β and γ are hyper-parameters utilized in our model. Importantly, it's worth noting that the MPL and PAL modules are trained collaboratively. During the inference stage, the test samples will only traverse the feature extractors and the predictor to acquire abnormal scores, and the MPL and PAL modules incur no additional computational cost.

## D. Inference Process

The inference process is identical to the baseline model, i.e. TEVAD [12], which is the left part of Figure 2 without the prompt dictionary. We initially extract visual and text features, which are processed through the feature alignment and fusion operation. Then, the fused features are fed into the anomaly predictor to calculate the anomaly score for each video snippet.

## IV. EXPERIMENTS

In this section, the performance of our LAP model is evaluated on four datasets, namely XD-Violence [39], UCFCrime [10], TAD [40] and ShanghaiTech [41]. The area under the precision-recall curve, also known as the average precision (AP) is employed as the evaluation metric for XD-Violence following the protocol in [35]. For UCF-Crime, TAD and ShanghaiTech, the area under the curve (AUC) of the framelevel receiver operating characteristics (ROC) is used instead. Specifically, AUCall represents the AUC for all testing videos, while AUCabn focuses only on abnormal videos in test set. The

TABLE II PERFORMANCE COMPARISON OF STATE -OF -THE -ART METHODS ON TAD (AUC%) AND SHANGHAITECH (AUC%). BOLD AND UNDERLINE INDICATE THE BEST AND SECOND -BEST RESULTS .

| Type     | Source    | Method          | Feat.    | TAD  AUC    | ST AUC   |
|----------|-----------|-----------------|----------|--------------|-----------|
| T        | Source    | Method          | Feat.    | TAD  AUC    | ST AUC   |
| mi       | ICCV 17’  | Luo et al. [43] | -        | 57.9         | -         |
| Sem      | CVPR 18’  | Liu et al. [44] | -        | 69.1         | 72.8      |
|          | CVPR 21’  | MIST [45]       | UNet     | 89.2         | 94.8      |
| ICCV 21’ | ICCV 21’  | RTFM [11]       | I3D      | 89.6         | 97.2      |
| TIP 2    | P 21’ W   | WSAL [40]       | I3D      | 89.6         | -         |
| CVPR 2   | CVPR 23’  | ECUPL [35]      | I3D      | 91.6         | -         |
| CVP      | CVPR 23’  | CMRL [36]       | I3D      | -            | 97.6      |
| CV       | CVPR 23’  | TEVAD [12]      | CLIP     | 92.3         | 97.3      |
|          | CVPR 23’  | UMIL [28]       | XCLIP    | 92.9         | 96.8      |
|          | ours      | LAP             | CLIP     | 94.4         | 97.4      |

false alarm rates for all videos (FARall) and abnormal videos (FARabn) are also reported in our ablation studies.

## A. Datasets

XD-Violence [39] is a multi-scene public dataset for VAD. It consists of a total duration of 217 hours and includes 4,754 untrimmed videos. The training set contains 3,954 videos while the test set comprises 800 videos. XD-Violence covers various unusual types of events including abuse incidents, car accidents, explosions, fights, riots, and shootings. UCFCrime dataset [10] is a large-scale collection of 1,900 videos captured by surveillance cameras in various indoor and outdoor scenarios. This dataset consists of 1,610 labeled training videos and 290 labeled test videos with a total duration of 217 hours. The dataset covers 13 types of anomalous events such as abuse, robbery, shootings and arson. TAD [40] is a dataset for anomaly detection in traffic scenes, consisting of 400 training videos and 100 test videos, with a total of 25 hours of video footage. It covers seven types of real-world anomalies. ShanghaiTech consists of surveillance videos from different scenes on a campus [42]. The training set contains 237 videos while the testing set has 200 videos.

## B. Implementation Details

The dimension of the visual features d v extracted by CLIP(ViT-L/14) [15] is 768, while the dimension of semantic features d t is also 768. The prompt dictionary capacity P is set to 30 for UCF-Crime, XD-Violence and TAD datasets, 25 for ShanghaiTech. The batch size b is set to 64 for TAD dataset, and it is halved to 32 on the other three datasets. The number of snippets per video L is set to 64 for all datasets. The feature operation θ is set as, a) concatenation for UCF-Crime, b) addition for the other three datasets. The hyper-parameters α = 1 , β = 0 . 1 , γ = 0 . 001 and τ = 1 are consistent across all datasets. The Adam optimizer is utilized with a learning rate of 0.001 and weight decay of 0.005 during the training process.

Fig. 4. Qualitative comparisons of TEVAD [12] and our method on both UCF-Crime (UCF) and XD-Violence (XD). The ground truth of anomalous events is represented by light red regions.

![Image](artifacts/image_000003_8d6eec2730f35ce077300e848ce36f82848425f5e9723beca304ea398f837695.png)

## C. Comparison Results

Quantitative analysis. The comparisons between our LAP model and other state-of-the-art (SOTA) WS-VAD models on the XD-Violence and UCF-Crime datasets are presented in Table I, and Table II shows the comparisons on the TAD and Shanghaitech datasets. It can be seen that the proposed model outperforms almost all the other methods in all datasets.

Specifically, our model achieves the highest AP of 86.5% on the XD-Violence dataset, outperforming the second best method VadCLIP [13] by 2.0%, which also combines RGB and text data. Unlike the single-word description, i.e. class labels, used in VadCLIP, the event-level descriptions in our LAP model can provide much richer information, leading to a better understanding of the anomalies. Another ClIP-based method (CLIP-TSA [23]) leverages the visual features from CLIP (VIT/B), while a transformer is employed to enhance its features. However, due to the lack of semantic guidance, its AP falls 4.3% below the proposed LAP. The performance of the other compared the methods are also limited by the absence of efficient anomaly definitions.

In the UCF-Crime dataset, our LAP model achieves an AUCall of 88.9%, surpassing the most recent methods by at least 0.9%, including VadCLIP [13] (88.0%), CLIP-TSA [23] (87.6%) and UMIL [12] (86.7%). Notably, the learnable prompts of VadCLIP are based on class labels, which leads to a video-level anomaly matching. While our concise descriptions of basic suspected anomaly events can effectively match the segment-level features. This difference results in a relatively high AUCabn of our LAP (73.0%) comparing to VadCLIP (70.2%). It is important to note that if we use more accurate text descriptions [46] of each snippet, we can achieve a higher AUCall of 90.4% and an AUCabn of 76.1%. The results indicate that our model can effectively utilize the textual prompts of abnormal events for an accurate detection.

Table II shows the comparisons on the other two less challenging datasets. The AUC of our approach (94.4%) is constantly higher than all SOTA methods compared [35], [40], [28] by a margin of 1.5% to 4.8% on TAD dataset. And our method achieves the second highest AUC (97.4%) in ShanghaiTech, which is only 0.2% lower than the best CMRL method [36]. Noting that, if we switch our visual extractor from CLIP to I3D [8] as the same as CMRL, the AUC will be boosted to 98.0% (0.4% higher CMRL). It indicates that the ShanghaiTech dataset is relatively less complex, while I3D [8] is good enough. The details will be discussed in Section IV-E. For fair comparation, we reimplement TEVAD[12] with visual features from CLIP. The AUC of our LAP exceeds TEVAD for 2.1% in the TAD dataset and 0.1% in the ShanghaiTech dataset. This minor performance gain on ShanghaiTech is due to the anomalies on campus being actually common activities such as riding, skating, and driving on the road, which are quite different from our suspected anomaly descriptions such as fighting, firing, or clashing.

Overall, these results highlight the superior performance of our LAP model compared to state-of-the-art methods on all four datasets in terms of both AP and AUC metrics. For fair comparison, the UMIL [28], TEVAD [12], CLIP-TSA [23], VadCLIP [13], and the work by Wu et al. [38] are based on the same feature extractor (CLIP) as our LAP.

Fig. 5. The distribution of matched suspected anomalies in the UCF-Crime (upper) and TAD (lower) datasets.

![Image](artifacts/image_000004_d7c6a55d3532a7fc913aaad96eed6aac33a45de87c6e9941bc63af08c25027c8.png)

Qualitative analysis. To further demonstrate the effectiveness of our method, the qualitative comparisons between our approach (LAP) and the TEVAD SOTA method [12] are visualized in Figure 4. The normal and abnormal frames of videos from the UCF-Crime and XD-Violence datasets are presented along with their corresponding frame-level anomaly scores, while green and red dashed rectangles indicate normal and abnormal ones, respectively. As shown in the figures, our method not only outperforms TEVAD [12] in terms of anomaly detection ability but also reduces false alarms on normal parts.

Our prompt dictionary contains event descriptions for various conditions. For the UCF-Crime social dataset and the TAD traffic dataset, Figure 5 illustrates the distributions of matched suspected anomalies in the anomaly vector c, showcasing how our prompt dictionary operates. Since the majority of anomalies in UCF-Crime are linked to human behavior like fights, robbery and violence, the predominant subject is "person" and the most frequent activities include falling and using weapons. While in the other circumstance, the Traffic Anomaly Dataset (TAD) is consists of anomalies caused by traffic accidents. As expected, "car" is the main subject, and the most common activities involve smoking and crashes. It indicates the effectiveness of our proposed event prompts.

## D. Ablation Studies

Components. The proposed prompt-related components, i.e. feature synthesis (FS), multi-prompt learning (MPL) and pseudo anomaly labeling (PAL) are the keys to our superior performance in VAD. The ablation results of these three components on three datasets are shown in Table III. The baseline module is a visual-only branch MIL-based network with a CLIP feature extractor [15]. By cooperating text branch for feature synthesis, our model achieves 1.2%, 1.8% and 0.4% AUCall improvement, respectively, which shows the efficiency of semantic information. The MPL module can also improve the AUCall for all datasets by 0.3%, 0.4% and 0.1%, while

TABLE III ABLATION STUDY OF PROPOSED MODULES. THE DEFAULT SETTINGS OF ALL EXPERIMENTS ARE MARKED IN GRAY COLOR .

| ne    |    |     |     | UCF-Crime    | UCF-Crime    | XD-Violence    | XD-Violence    | TAD      |
|-------|----|-----|-----|--------------|--------------|----------------|----------------|----------|
| Basel | FS | MPL | PAL | AUC  ll b   | AUC  ll b   | AUC  all      | AP  all       | AUC all |
| ✓     |    |     |     | 87.0         | 67.0         | 93.2           | 81.3           | 93.7     |
| ✓     | ✓  |     |     | 88.2         | 70.4         | 95.0           | 84.1           | 94.1     |
| ✓     | ✓  | ✓   |     | 88.5         | 70.7         | 95.4           | 85.0           | 94.2     |
| ✓     | ✓  | ✓   | ✓   | 88.9         | 73.0         | 95.6           | 86.5           | 94.4     |

TABLE IV COMPARISONS OF THE AUC (%) FOR OPEN-SET VAD ON UCF-CRIME . THE NUMBERS IN BRACES ARE THE AMOUNT OF VIDEOS .

| Open  Category    |   No  | Explo-  sion (21)    | RoadAcci-  dents (23)    | Shoplif ting (21)   |
|--------------------|-------|-----------------------|---------------------------|-----------------------|
| RTFM [11]          |  84.3 | 83.6(-0.7)            | 82.1(-2.2)                | 83.4(-0.9)            |
| MLAD [47]          |  85.4 | 84.3(-1.1)            | 83.2(-2.2)                | 84.5(-0.9)            |
| TEVAD [12]         |  84.9 | 83.7(-1.2)            | 81.0(-3.9)                | 83.1(-1.8)            |
| Ours               |  88.9 | 88.1(-0.8)            | 87.0(-1.9)                | 88.4(-0.5)            |

the AP on XD-Violence and AUCabn on UCF-Crime are also boosted by 0.9% and 0.3%. Further incorporating the PAL module yields better results. It outperforms the baseline by 1.9% in AUCall and 6.0% in AUCabn on UCF-Crime, 0.7% in AUCall on TAD, as well as 2.4% in AUCall and 5.2% in AP on XD-Violence.

Prompt format. The prompt format plays an important role in the proposed LAP model. Thus, two different formats are tested in this experiment. One is organized by anomaly phrases, such as "falling down" or "on fire". The other contains complete anomaly sentences, like "someone is doing something to whom" or "something is what". As shown in Figure 6a, the sentence-based prompt dictionary outperforms the phrase-based one by 3.1% on XD-Violence and 0.7% on UCF-Crime, respectively. It suggests that prompts containing richer information are more helpful in identifying suspected anomalies.

Pseudo anomaly threshold. As required by the PAL module, pseudo labels are determined according to the threshold Gh. The dynamic threshold given in Eq. 13 is used in previous experiments, which is based on the distribution of the data in the current batch. The hyper-parameter τ will determine the number for pseudo anomalies. When it is set to 0.5, 1.0 and 2.0, the AUC results on UCF-Crime are 88.05%, 88.90%, and 88.21%, correspondingly. Another static threshold strategy is also compared in this test, while Gh is set to 0.5 as prior knowledge. As shown in Figure 6b, the dynamic threshold is better than the static one, whose AP and AUC are 2.1% and 0.5% higher on XD-Violence and UCF-Crime datasets.

TABLE V CROSS -DATASET EXPERIMENTAL RESULTS ON UCF-CRIME (UCF) AND XD-VIOLENCE (XD) BENCHMARKS .

| Source    | UCF    | XD           | XD    | UCF          |
|-----------|--------|--------------|-------|--------------|
| Target    | UCF    | (AUC %)      | XD    | XD (AP %)    |
| RTFM [11] | 84.3   | 68.6 (-15.7) | 76.6  | 37.3 (-39.3) |
| CMRL [36] | 6.1    | 69.9 (-16.2) | 81.3  | 46.7 (-34.6) |
| Ours      | 88.9   | 83.5 (-5.4)  | 86.5  | 60.9(-25.6)  |

Fig. 6. The ablation studies of the prompt format and pseudo anomaly threshold.

![Image](artifacts/image_000005_f114e2f5a51324ab722fe0bd2bb7968a94886e22cd4f32f69627419f3e0ac061.png)

## E. Discussions

Class-wise AUC. To demonstrate the detailed performance on specific abnormal events, the class AUC of our model is compared with RTFM [11] in Figure 7. It shows that the proposed LAP model outperforms RTFM in most categories, especially on "Assault", "Explosion", "RoadAccidents" and "Robbery". This can be attributed to the effective use of our prompt dictionary to describe those anomalies including representative texts such as "fire", "knife" or "accident". Combined with the MPL module, their synthetic features are more likely to be identified as abnormal ones. However, our model may be less effective in some cases if the action is subtle or difficult to describe, such as "Shoplifting" and "Fighting" in Figure 7.

Open set VAD. In practical applications, it is impossible to collect or define all possible anomalies in advance. Hence, it is crucial to examine the robustness of anomaly detection models when confronted with open abnormal categories in real-world scenarios. Following the protocol of open set VAD in MLAD [47], experiments are conducted on the top 3 largest anomaly categories from the UCF-Crime dataset, namely "Explosion", "RoadAccidents" and "Shoplifting". These categories are sequentially removed from the training set and treated as real open abnormal events. The comparisons with three SOTA models are presented in Table IV. It is obvious that the proposed LAP model outperforms RTFM [11], MLAD [47] and TEVAD [12] in all three categories. It is worth noting that our method achieves minimal decreases in AUC values when compared to alternative approaches. This indicates that our method is more efficient in handling open abnormal event issues.

Cross-dataset performance. The categories of anomalies varies from different VAD datasets. For instance, the abnormal events in UCF-Crime dataset are collected from surveillance videos, which is quite different from the abnormal categories

TABLE VI PERFORMANCE OF MPL AND PAL EMBEDDED RTFM [11] ON SHANGHAITECH .

| Method (ST)    |   AUCall  |   AUCabn  |   FARall  |   FARabn |
|----------------|-----------|-----------|-----------|----------|
| RTFM           |      97.2 |      64.3 |      0.06 |     0.86 |
| RTFM+MPL       |      97.6 |      72.2 |      0.06 |     0.71 |
| RTFM+PAL       |      97.5 |      73.9 |      0.03 |     0.44 |
| RTFM+Both      |      98   |      75.6 |      0.04 |     0.58 |

Fig. 7. Comparison of class-wise AUC (%) on UCF-Crime dataset with RTFM [11].

![Image](artifacts/image_000006_2a7e080a7fc374b2f3a0ac011d040963e4f730dde315af3edf076c41b935252a.png)

in XD-Violence developed from movies and online videos. Thus, it will become a challenging transfer learning task, if the model is trained and inferred on different datasets. However, it is actually what will happen in real-world anomaly detection applications. To evaluate the generalization and zero-shot abilities of our proposed method, another set of experiments using different sources of training and inference videos is conducted. Compared with RTFM [11] and CMRL [36], our model explains the definition of anomalies with their descriptions using the prompt dictionary and multi-prompt learning scheme. Such a new paradigm of utilizing the semantic information leads to an extraordinary cross-dataset performance as shown in Table V. The performance degradation of the proposed model is only one-third of the ones of RTFM and CMRL, when it is trained on XD-Violence and tested on ShanghaiTech. It indicates that our method is much less sensitive to variations in the data domain, which is important for practical applications.

Plug and play. To further explore the potential of our method, the proposed MPL and PAL modules are embedded into the representative WS-VAD work RTFM [11] on ShanghaiTech. For a fair comparison, I3D [8], instead of CLIP [15], is used as the feature extractor, while all experimental settings were kept the same as in RTFM paper. As shown in Table VI, the reimplemented frameworks generally exhibited better performance. By incorporating either MPL or PAL alone, enhancements of 0.4% and 0.3% can be achieved on AUCall, whereas more significant enhancements of 7.9% and 9.6% can be observed on AUCabn. The efficacy of MPL and PAL is demonstrated by their ability to improve the performance of the conventional WSAD framework. Through the collaboration of MPL and PAL, LAP integrated RTFM demonstrates superior AUC and reduced FAR compared to

its original version, showing significant enhancements (0.8%, 0.02%) on the ShanghaiTech dataset. It is worth noting that the reimplemented RTFM model (98. 0%) could even surpass the latest SOTA model CMRL [36] by 0.4%. It indicates that more frameworks may benefit from our prompt-related modules, which are plug-and-play.

## V. CONCLUSION

In this study, we presented the LAP model, a straightforward yet effective method for WS-VAD. Specifically, the synthesized visual-semantic features have been employed for better feature representation. The multi-prompt learning strategy has shown its capability to guide the learning of suspected anomalies with a prompt dictionary. Additionally, the pseudo anomaly labels generated by the anomaly similarity between the prompts and video captions are useful to enhance the VAD performance. Extensive experiments have demonstrated the effectiveness of our model. We hope that our work will inspire further exploration of defining and learning anomalies from natural languages.

## REFERENCES

- [1] Y. Zhao, B. Deng, C. Shen, Y. Liu, H. Lu, and X.-S. Hua, "Spatiotemporal autoencoder for video anomaly detection," in Proceedings of the 25th ACM international conference on Multimedia, 2017, pp. 1933– 1941.
- [2] G. Yu, S. Wang, Z. Cai, E. Zhu, C. Xu, J. Yin, and M. Kloft, "Cloze test helps: Effective video anomaly detection via learning to complete video events," in Proceedings of the 28th ACM international conference on multimedia, 2020, pp. 583–591.
- [3] C. Tao, C. Wang, S. Lin, S. Cai, D. Li, and J. Qian, "Feature reconstruction with disruption for unsupervised video anomaly detection," IEEE Transactions on Multimedia, 2024.
- [4] S. Yu, C. Wang, Q. Mao, Y. Li, and J. Wu, "Cross-epoch learning for weakly supervised anomaly detection in surveillance videos," IEEE Signal Processing Letters, vol. 28, pp. 2137–2141, 2021.
- [5] H. Shi, L. Wang, S. Zhou, G. Hua, and W. Tang, "Abnormal ratios guided multi-phase self-training for weakly-supervised video anomaly detection," IEEE Transactions on Multimedia, 2023.
- [6] J. Yu, B. Zhang, Q. Li, H. Chen, and Z. Teng, "Hierarchical reasoning network with contrastive learning for few-shot human-object interaction recognition," in Proceedings of the 31st ACM International Conference on Multimedia, ser. MM '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 4260–4268. [Online]. Available: https://doi.org/10.1145/3581783.3612311
- [7] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3d convolutional networks," in Proceedings of the IEEE international conference on computer vision, 2015, pp. 4489–4497.
- [8] J. Carreira and A. Zisserman, "Quo vadis, action recognition? a new model and the kinetics dataset," in proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.
- [9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al. , "An image is worth 16x16 words: Transformers for image recognition at scale," arXiv preprint arXiv:2010.11929, 2020.
- [10] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6479–6488.
- [11] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro, "Weakly-supervised video anomaly detection with robust temporal feature magnitude learning," in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 4975–4986.
- [12] W. Chen, K. T. Ma, Z. J. Yew, M. Hur, and D. A.-A. Khoo, "Tevad: Improved video anomaly detection with captions," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 5548–5558.
- [13] P. Wu etal., "Vadclip: Adapting vision-language models for weakly supervised video anomaly detection," AAAI, 2024.
- [14] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., "Training language models to follow instructions with human feedback," Advances in Neural Information Processing Systems, vol. 35, pp. 27 730–27 744, 2022.
- [15] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [16] H. Zhou, J. Yu, and W. Yang, "Dual memory units with uncertainty regulation for weakly supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 3, 2023, pp. 3769–3777.
- [17] X. L. Li and P. Liang, "Prefix-tuning: Optimizing continuous prompts for generation," arXiv preprint arXiv:2101.00190, 2021.
- [18] B. Lester, R. Al-Rfou, and N. Constant, "The power of scale for parameter-efficient prompt tuning," arXiv preprint arXiv:2104.08691 , 2021.
- [19] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim, "Visual prompt tuning," in European Conference on Computer Vision. Springer, 2022, pp. 709–727.
- [20] F. Sato, R. Hachiuma, and T. Sekii, "Prompt-guided zero-shot anomaly action recognition using pretrained deep skeleton features," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6471–6480.
- [21] X. Wang, R. Xian, T. Guan, and D. Manocha, "Prompt learning for action recognition," arXiv preprint arXiv:2305.12437, 2023.
- [22] Z. Liu, X.-M. Wu, D. Zheng, K.-Y. Lin, and W.-S. Zheng, "Generating anomalies for video anomaly detection with prompt-based feature mapping," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24 500–24 510.
- [23] Z. Joo etal., "Clip-tsa:clip-assisted temporal self-attention for weakly supervised video anomaly detection," ICIP, 2023.
- [24] C. Cao, X. Zhang, S. Zhang, P. Wang, and Y. Zhang, "Weakly supervised video anomaly detection based on cross-batch clustering guidance," in 2023 IEEE International Conference on Multimedia and Expo (ICME) . IEEE, 2023, pp. 2723–2728.
- [25] K. Lin, L. Li, C.-C. Lin, F. Ahmed, Z. Gan, Z. Liu, Y. Lu, and L. Wang, "Swinbert: End-to-end transformers with sparse attention for video captioning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17 949–17 958.
- [26] T. Gao, X. Yao, and D. Chen, "Simcse: Simple contrastive learning of sentence embeddings," arXiv preprint arXiv:2104.08821, 2021.
- [27] M.-C. Popescu, V. E. Balas, L. Perescu-Popescu, and N. Mastorakis, "Multilayer perceptron and neural networks," WSEAS Transactions on Circuits and Systems, vol. 8, no. 7, pp. 579–588, 2009.
- [28] H. Lv, Z. Yue, Q. Sun, B. Luo, Z. Cui, and H. Zhang, "Unbiased multiple instance learning for weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8022–8031.
- [29] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, "Learning temporal regularity in video sequences," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 733–742.
- [30] M. Z. Zaheer, A. Mahmood, M. H. Khan, M. Segu, F. Yu, and S.-I. Lee, "Generative cooperative learning for unsupervised video anomaly detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 14 744–14 754.
- [31] S. Li, F. Liu, and L. Jiao, "Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 2, 2022, pp. 1395–1403.
- [32] A. Panariello, A. Porrello, S. Calderara, and R. Cucchiara, "Consistencybased self-supervised learning for temporal anomaly localization," in European Conference on Computer Vision. Springer, 2022, pp. 338– 349.
- [33] H. Sapkota and Q. Yu, "Bayesian nonparametric submodular video partition for robust anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3212–3221.
- [34] Z. Yang, Y. Guo, J. Wang, D. Huang, X. Bao, and Y. Wang, "Towards video anomaly detection in the real world: A binarization embedded weakly-supervised network," IEEE Transactions on Circuits and Systems for Video Technology, 2023.
- [35] C. Zhang, G. Li, Y. Qi, S. Wang, L. Qing, Q. Huang, and M.-H. Yang, "Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection," in Proceedings of the IEEE/CVF

- Conference on Computer Vision and Pattern Recognition, 2023, pp. 16 271–16 280.
- [36] M. Cho, M. Kim, S. Hwang, C. Park, K. Lee, and S. Lee, "Look around for anomalies: Weakly-supervised anomaly detection via context-motion relational learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 137–12 146.
- [37] Y. Chen, Z. Liu, B. Zhang, W. Fok, X. Qi, and Y.-C. Wu, "Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 1, 2023, pp. 387–395.
- [38] P. Wu, X. Zhou, G. Pang, Y. Sun, J. Liu, P. Wang, and Y. Zhang, "Openvocabulary video anomaly detection," arXiv preprint arXiv:2311.07042 , 2023.
- [39] P. Wu, J. Liu, Y. Shi, Y. Sun, F. Shao, Z. Wu, and Z. Yang, "Not only look, but also listen: Learning multimodal violence detection under weak supervision," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16. Springer, 2020, pp. 322–339.
- [40] H. Lv, C. Zhou, Z. Cui, C. Xu, Y. Li, and J. Yang, "Localizing anomalies from weakly-labeled videos," IEEE transactions on image processing , vol. 30, pp. 4505–4515, 2021.
- [41] J.-X. Zhong, N. Li, W. Kong, S. Liu, T. H. Li, and G. Li, "Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 1237–1246.
- [42] Y. Zhang, H. Lu, L. Zhang, X. Ruan, and S. Sakai, "Video anomaly detection based on locality sensitive hashing filters," Pattern Recognition , vol. 59, pp. 302–311, 2016.
- [43] W. Luo, W. Liu, and S. Gao, "Remembering history with convolutional lstm for anomaly detection," in 2017 IEEE International conference on multimedia and expo (ICME). IEEE, 2017, pp. 439–444.
- [44] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly detection–a new baseline," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6536– 6545.
- [45] J.-C. Feng, F.-T. Hong, and W.-S. Zheng, "Mist: Multiple instance selftraining framework for video anomaly detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14 009–14 018.
- [46] T. Yuan, X. Zhang, K. Liu, B. Liu, C. Chen, J. Jin, and Z. Jiao, "Towards surveillance video-and-language understanding: New dataset baselines and challenges," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 052–22 061.
- [47] C. Zhang, G. Li, Q. Xu, X. Zhang, L. Su, and Q. Huang, "Weakly supervised anomaly detection in videos considering the openness of events," IEEE transactions on intelligent transportation systems, vol. 23, no. 11, pp. 21 687–21 699, 2022.