## Towards Video Anomaly Retrieval from Video Anomaly Detection: New Benchmarks and Model

Peng Wu, Jing Liu Senior Member, IEEE, Xiangteng He, Yuxin Peng Senior Member, IEEE , Peng Wang, and Yanning Zhang Senior Member, IEEE

Abstract—Video anomaly detection (VAD) has been paid increasing attention due to its potential applications, its current dominant tasks focus on online detecting anomalies, which can be roughly interpreted as the binary or multiple event classification. However, such a setup that builds relationships between complicated anomalous events and single labels, e.g., "vandalism", is superficial, since single labels are deficient to characterize anomalous events. In reality, users tend to search a specific video rather than a series of approximate videos. Therefore, retrieving anomalous events using detailed descriptions is practical and positive but few researches focus on this. In this context, we propose a novel task called Video Anomaly Retrieval (VAR), which aims to pragmatically retrieve relevant anomalous videos by cross-modalities, e.g., language descriptions and synchronous audios. Unlike the current video retrieval where videos are assumed to be temporally well-trimmed with short duration, VAR is devised to retrieve long untrimmed videos which may be partially relevant to the given query. To achieve this, we present two large-scale VAR benchmarks and design a model called Anomaly-Led Alignment Network (ALAN) for VAR. In ALAN, we propose an anomaly-led sampling to focus on key segments in long untrimmed videos. Then, we introduce an efficient pretext task to enhance semantic associations between video-text finegrained representations. Besides, we leverage two complementary alignments to further match cross-modal contents. Experimental results on two benchmarks reveal the challenges of VAR task and also demonstrate the advantages of our tailored method. Captions are publicly released at https://github.com/Roc-Ng/VAR.

Index Terms—video anomaly retrieval, video anomaly detection, cross-modal retrieval

## I. INTRODUCTION

V IDEO anomaly detection (VAD) plays a critical role in video content analysis, and has become a hot topic being studied due to its potential applications, e.g., danger earlywarning. VAD, by definition, aims to identify the location of anomaly occurrence, which can be regarded as the framelevel event classification. VAD can be broadly divided into two categories, i.e., semi-supervised [1]–[5] and weakly supervised [6]–[10]. The former typically recognizes anomalies

Peng Wu, Peng Wang, and Yanning Zhang are with the National Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big Data Application Technology, School of Computer Science, Northwestern Polytechnical University, China. E-mail: xdwupeng@gmail.com; peng.wang, ynzhang@nwpu.edu.cn. Jing Liu is with the Guangzhou Institute of Technology, Xidian University, China. E-mail: neouma@163.com. Xiangteng He and Yuxin Peng are with the Wangxuan Institute of Computer Technology, Peking University, China. E-mail: hexiangteng, pengyuxin@pku.edu.cn. This work is supported by the National Natural Science Foundation of China (No. 62306240, U23B2013, U19B2037, 62272013, 61925201, 62132001), China Postdoctoral Science Foundation (No. 2023TQ0272), and the Fundamental Research Funds for the Central Universities (No. D5000220431). (Corresponding author: Yanning Zhang.)

Manuscript received April 19, 2021; revised August 16, 2021.

Fig. 1. VAD vs. VAR. Single labels may be unable to describe sequential anomalous events in VAD, but text captions or synchronous audios can sufficiently depict events in VAR.

![Image](artifacts/image_000000_74c2cccd8aae7c04ae69d587bf00568414a23b6590f5c0af395ace1ca5cdfb7f.png)

through self-supervised learning or one-class learning. The latter, thanks to massive normal and abnormal videos with video-level labels, achieves better detection accuracy.

Impressive progress has been witnessed for VAD, however, an event in videos generally captures an interaction between actions and entities that evolves over time, simply utilizing single labels in VAD may be insufficient to explain the sequential events depicted. Besides, compared with VAD, offline video search thus far is still more commonly used in realworld applications. Imagining the case when searching for related videos, we prefer to use comprehensive descriptions to accurately search, e.g., "At night, two topless men smashed the door of the store.", rather than use a single coarse word, e.g, "vandalism", to get a large collection of rough results.

Based on VAD, we propose a new task called Video Anomaly Retrieval (VAR) and present two large-scale benchmark datasets, UCFCrime-AR and XDViolence-AR, to further facilitate the research of video anomaly analysis. The goal of VAR is to retrieve relevant untrimmed videos given crossmodal queries, e.g., text captions and synchronous audios, and vice versa. Unlike VAD, VAR depicts anomalies from multiple viewpoints and sufficiently characterizes sequential events. We illustrate the advantage of video anomaly retrieval in Figure 1. VAR task has high value to real-world applications, especially for smart ground and car surveillance. Generally speaking, for surveillance, the recorded video will be stored in the hard disk or memory card as a series of segments with a certain time length. After an abnormal event occurs, we need to search the corresponding video segment that contains the queried abnormal event through the descriptions, such as a white car crashed into the rear of a van, a group of people breaking into a house at night, etc.

Our VAR is considerably different from traditional video retrieval (VR) [11]–[13]. In traditional video retrieval, videos

Query: A girl playing guitar and singing a song

.

Query: An adult brown horse stand in the barn and his father horse jumps a barrier and his mother

Fig. 2. Comparison of VAR with video retrieval and video moment retrieval.

![Image](artifacts/image_000001_78c2271e70fb37554c7b83edba4dbacb7b615b86faa4fd44b076535d9de6be9b.png)

are assumed to be temporally pre-trimmed with short duration, and thus the whole video is supposed to be completely relevant to the paired query. In reality, videos are usually not welltrimmed and may only exist partial fragments to fully meet the query. In VAR, the main goal is to retrieve long and untrimmed videos. Such a setup more meets realistic requirements, and also evokes the new challenge. Concretely, the length of relevant fragments is variable in videos w.r.t. given paired queries. For normal videos, the relevant fragment is generally the whole video; For abnormal videos, the relevant fragment may occupy only a fraction or a lion's share of the entire video since the length of anomalous events is inconstant in videos. Besides, our VAR task also differs from video moment retrieval (VMR) [14]–[17] since the latter is to retrieve moments rather than untrimmed videos. Because both abnormal videos and normal videos (no anomaly) need to be retrieved in VAR, video moment retrieval methods are hard to tackle this task. Traditional video retrieval and video moment retrieval methods cannot solve this new challenge well, detailed results are listed in Tables II and III. The differences between video retrieval, video moment retrieval and video anomaly retrieval are shown in Figure 2.

To overcome the above challenge, we propose ALAN, an Anomaly-Led Alignment Network for video anomaly retrieval. In ALAN, video, audio, and text encoders are intended to encode raw data into high-level representations, and crossmodal alignment is introduced to match cross-modal representations from different perspectives. Since videos are long untrimmed and anomalous events have complex variations in the scenario and length, we expect that, in the encoding phase, the retrieval system maintains holistic views, meanwhile, focuses on key anomalous segments, so that cross-modal representations can be well aligned in the joint embedding space. Therefore, vanilla fixed-frame sampling, e.g., uniform sampling and random sampling, is not flexible to focus on specific anomalous segments. Inspired by dynamic neural networks [18]–[21], we propose an anomaly-led sampling, which simply resorts to frame-level anomaly priors generated by an ad-hoc anomaly detector and does not require intensive pairwise interactions between cross modality, to select key segments with large anomaly identification degree. We then

Query: two teams playing volleyball

.

couple these two win-win sampling mechanisms for videos and audios, where anomaly-led sampling focuses on anomalous segments, and fixed-frame sampling pays attention to the entirety as well as normal videos. Furthermore, to establish associations between video-text fine-grained representations as well as maintain high retrieval efficiency, we also propose a pretext task, i.e., video prompt based masked phrase modeling (VPMPM), serving the model training. Particularly, a new module termed Prompting Decoder takes both frame-level video representations and contextual text representations as input and predicts the masked noun phrases or verb phrases by the cross-modal attention, where video representations serve as fixed prompts [22], [23]. In this paper, video frames are regarded as the fine granularity as frames usually reflect more detailed content of videos, meanwhile noun phrases and verb phrases in texts, e.g., "a black car" and "left quickly", are regarded as the fine granularity, which reflect the local spatial contents and temporal dynamics in the video, respectively. Notably, compared with nouns and verbs, noun phrases and verb phrases contain more contents, and can also particularly illustrate the subtle differences. Finally, such a proxy training objective optimizes the encoder parameters and further promotes the semantic associations between local video frames and text phrases by cross-modal interactions.

To summarize, our contributions are three-fold:

- We introduce a new task named video anomaly retrieval to bridge the gap between the literature and real-world applications in terms of video anomaly analysis. To our knowledge, this is the first work moves towards VAR from VAD;
- We present two large-scale benchmarks, i.e., UCFCrimeAR and XDViolence-AR, based on public VAD datasets. The former is applied to video-text VAR, the latter is to video-audio VAR;
- We propose a model called ALAN, aiming at challenges in VAR, where anomaly-led sampling, video prompt based masked phrase modeling, and cross-modal alignment are introduced for the attention of anomalous segments, enhancement of fine-grained associations, and multi-perspective match, respectively.

## II. RELATED WORK

## A. Video Anomaly Detection

Aided by the success of deep learning, VAD has made much good progress in recent years, which is usually classified into two groups, semi-supervised anomaly detection and weakly supervised anomaly detection. In semi-supervised anomaly detection, only normal event samples are available for model training. Recent researchers mainly adopt deep auto-encoders [2], [4], [24]–[30] for self-supervised learning, e.g., reconstruction, prediction, jigsaw, etc. Weakly supervised anomaly detection can be regarded as the problem of binary classification, to obtain frame-level predictions given coarse labels, and multiple instance learning [6], [8], [9], [31]–[34] is widely used to train models. Unlike VAD that utilizes single labels to distinguish whether each frame is anomalous or not, our proposed VAR uses elaborate text descriptions or synchronous audios to depict the sequential events.

## B. Cross-Modal Retrieval

We mainly introduce cross-modal retrieval [35]–[38] built on videos, texts, and audios. There are some works [39]– [44] focus on audio-text and audio-video retrieval. Specifically, Tian et al. [45] propose an audio-to-video/video-toaudio cross-modality localization/retrieval task [46], i.e, given a sound segment, locate the corresponding visual sound source temporally within a video, and vice versa. Then Wu et al. [47] introduce a novel dual attention matching method for this task. Recently, Lin et al. [48] propose a latent audio-visual hybrid adapter that adapts pre-trained vision transformers to audiovisual tasks, this method focuses on audio-video event localization task rather than cross-modal retrieval. In addition, textvideo retrieval is a key role in cross-modal retrieval. Generally, text-video retrieval can be divided into two categories, i.e., dual-encoder and joint-encoder. Dual-encoder based methods usually train two individual encoders to learn video and text representations and then align these representations in joint embedding spaces. Among them, some works [13], [49]–[51] focus on learning single global representations, but they lack the consideration of fine-grained information. Thereby, several works devote efforts to aligning fine-grained information [52]– [58]. Joint-encoder based methods [59]–[62] typically feed the video and text into a joint encoder to capture their cross-modal interactions. In comparison to dual-encoder based methods, joint-encoder based methods explicitly learn finegrained associations and achieve more impressive results, but sacrifice the retrieval efficiency since every text-video pair needs to be fed into the encoder at the inference time.

Different from the above video retrieval, we consider a more realistic scenario, where most videos contain anomalous events, and a more realistic demand, where videos are long untrimmed and partially relevant to the cross-modal query [63]. Such a new task poses extra challenges as well as multi-field research points. In addition, our ALAN also differs from video moment retrieval methods [64]–[67] in that it does not require complex cross-modal interactions.

## III. BENCHMARK

Manually collecting a large-scale video benchmark is laborintensive and time-consuming, it is also subjective since video understanding can often be an ill-defined task with low annotator consistency [68]. Therefore, we start with two acknowledged datasets in VAD community, i.e., UCF-Crime [6] and XD-Violence [7], and construct our benchmarks for VAR. We adopt these two datasets as the base since they thus far are the two most comprehensive VAD datasets in terms of length and scene, where the total duration of them are 128 and 217 hours, respectively. Besides, they are also collected from a variety of scenarios. For example, UCF-Crime covers 13 realworld anomalies as well as normal activities, and XD-Violence captures 6 anomalies and normal activities from movies and YouTube. In addition, both of them contain half normal videos and half abnormal videos, therefore, retrieval systems retrieve both abnormal and normal videos from the video gallery given related cross-modal queries in VAR. Large and diverse video databases allow us to construct more practicable benchmarks for VAR.

## A. UCFCrime-AR

UCF-Crime dataset consists of 1900 untrimmed videos with 950 abnormal videos and 950 normal videos. Notably, for anomaly videos in the training set, the start timestamp and duration of anomalous activities are unavailable. For normal videos, they are totally anomaly-free. We directly use the total videos as the video search base. To achieve cross-modal retrieval, we require pairwise text descriptions.

We invite 8 experienced annotators who are proficient in Chinese and English to annotate these videos. The annotators watch the entire video and make the corresponding captions in both Chinese and English. Specifically, annotators are required to focus on anomalous events when describing anomaly videos. Due to the subtle differences in videos for the same anomaly category, we need to obtain quality sentence annotations to distinguish fine differences and avoid being into a one-to-many dilemma [69] which often appears in the current video retrieval. To be specific, there are at most two annotators to describe videos in the same category. For two similar videos in the same category, describe their differences in detail as much as possible. Take the scene of a fighting between two people as an example, e.g., "At a party, the yellow-haired man suddenly attacked a man opposite him.", "A young man suddenly beat another man with glasses in the elevator." The above two annotations clearly describe the difference between two similar videos. Finally, we double-check each sentence description to guarantee the quality.

Following the partition of UCF-Crime, UCFCrime-AR includes 1610 training videos and 290 test videos. Each video is annotated with captions in both English and Chinese. In this work, we only use captions in English.

## B. XDViolence-AR

As for XD-Violence, we found that it is very hard to describe videos in a few sentences due to their complicated contents and scenarios. Hence we changed focus and started a new line of audio-to-video retrieval due to its natural audiovisual information, that is, we use videos and synchronous audios for cross-modal anomaly retrieval. Unlike texts, audios have the same granularity as videos. Similar to UCF-Crime, XD-Violence is also a weakly supervised dataset, namely, frame-level annotations are unknown. XDViolence-AR is split into two subsets, with 3954 long videos for training and 800 for testing.

## C. Benchmark Statistics

We compare two benchmarks with several cross-modal retrieval/location datasets in Table I. As we can see that, video databases in UCFCrime-AR and XDViolence-AR are both large-scale and are made public in recent years, where the former is applied to video-text (V-T) anomaly retrieval, and the latter is applied to video-audio anomaly retrieval (VA). Notably, the average length of videos in VAR benchmarks is significantly longer than that of videos in traditional video retrieval datasets. For example, the average length of videos of UCFCrime-AR and XDViolence-AR are 242s and 164s,

TABLE I COMPARISON OF UCFCRIME-AR AND XDVIOLENCE-AR WITH SEVERAL VIDEO -TEXT AND VIDEO -AUDIO RETRIEVAL/LOCALIZATION DATASETS .

| Datasets       | Duration    | #Videos    | Avg.len.    | Type    |   Year |
|----------------|-------------|------------|-------------|---------|--------|
| MSR-VTT [70]   | 40h         | 7.2k       | 20s         | V-T     |   2016 |
| VATEX [71]     | 114h        | 41k        | 10s         | V-T     |   2019 |
| TVR [74]       | 463h        | 21.8k      | 76s         | V-T     |   2020 |
| AVE [45]       | 11h         | 4.1k       | 10s         | V-A     |   2018 |
| AudioCaps [72] | 127h        | 46k        | 10s         | V-A     |   2019 |
| LLP [73]       | 33h         | 11k        | 10s         | V-A     |   2020 |
| UCFCrime-AR    | 128h        | 1.9k       | 242s        | V-T     |   2018 |
| XDViolence-AR  | 217h        | 4.8k       | 164s        | V-A     |   2020 |

Fig. 3. Statistical histogram distributions on UCFCrime-AR. Left: text captions in English; Right: text captions in Chinese.

![Image](artifacts/image_000002_85994240b47a82186dbafe28b67e844fe2f333493dfba5c89231268514e565cd.png)

whereas that of MSR-VTT [70], VATEX [71], VAE [45], AudioCaps [72], and LLP [73] are in the range of 10s to 20s, and TVR [74] is mainly applied to video moment retrieval task, its average length of videos is still much shorter than our benchmarks. Longer videos emphasize again the goal of VAR is to retrieve long and untrimmed videos, such a setup meets realistic requirements, and also reveals VAR is a more challenging task. For video-text UCFCrime-AR, we also present histogram distributions of captions in Figure 3. The average caption lengths of UCFCrime-AR-en, UCFCrime-ARzh are 16.3 and 22.4, which are longer than those of previous datasets in video retrieval. e.g., the average caption lengths of VATEX-en [71], VATEX-zh [71], and MSR-VTT [70] are 15.23, 13.95, and 9.28, respectively.

## IV. METHOD

In this section, we introduce ALAN in detail. In Sec. IV-A, we first introduce three encoders in ALAN, namely, video encoder, text encoder, and audio encoder, the goal of these encoders is to project raw videos, texts, and audios into high-level representations. In Sec. IV-B, we introduce the anomaly-led sampling mechanism which is utilized in both video encoder and audio encoder. In Sec. IV-C, we describe a novel pretext task, i.e., VPMPM, which is applied to videotext anomaly retrieval. At last, we describe the cross-modal alignment and training objectives in Secs. IV-D and IV-E.

## A. Encoders

Video encoder. Unlike images, videos possess space-time information [75], [76]. As a consequence, we consider both appearance and motion information to encode videos. Specifically, given an video v, we use I3D-RGB and I3D-Flow pre-trained on Kinetics [77] to extract frame-level object and motion features, respectively, then project these features into a d-dimensional space for the subsequent operations. Here, object and motion feature sequences are denoted as F o (v) and F m (v), respectively. Both sequences contain T clip features. For the sake of clarity, we use F(v) to denote F o (v) and F m (v). Taking into account the variety of anomalous event duration in untrimmed videos, we sample two sparse video clips with different concerns, i.e., U and R, from F(v) by means of the fixed-frame sampling and our proposed anomalyled sampling.

As demonstrated in Figure 4, the video encoder is a symmetric two-stream model, one stream takes as input object, and the other takes as input motion. In order to fuse features in different modalities and different temporalities for final representations, we employ the Transformer [78] as the base model, which has been widely used in VAD and VR tasks with good results. For example, Huang et al. [34], [79] and Zhao et al. [80] used Transformer to tackle VAD and VR tasks, respectively. We first concatenate two different sampling clips as a new sequence, i.e., [UCLS, U1, ..., RN , RCLS, R1, ..., RN ] , where UCLS and RCLS are [CLS] tokens, which are the average aggregation of all features in U and R, respectively. Then, we add positional embeddings [78] and sequence embeddings to this sequence. Here positional embeddings provide temporal information about the time in the video, and sequence embeddings depict that features in U and R stem from different sequences. In video encoder, Self Encoder is devised to capture contextual information, which is a standard encoder layer in Transformer. The following Cross Encoder takes the selfmodality as the query, and cross-modality contextual features as the key and value to encode cross-modal representations through cross-modal attention. Cross Encoder is composed of multi-head attention, a linear layer, a residual connection, and a layer normalization. Finally, we obtain two different video representations, one is the average of output from UCLS and RCLS, denoted as g v (including g vo and g vm ), the other is the mean of average pooling aggregation of output from U and R, denoted as h v (including h vo and h vm ). Such a simple pooling operation is parameter-free and effective in our work, enabling h v to involve local fine-grained information.

Text encoder. Give a text caption t, we aim to learn the alignment between it and a related video at two different levels. At first, we leverage a pre-trained BERT [13] to extract features

Fig. 4. Overview of our ALAN. It consists of several components, i.e., video encoder, text encoder, audio encoder, pretext task VPMPM, and cross-modal alignment.

![Image](artifacts/image_000003_4cc93c367f5816af5df182bb8dc46c8bd4b37452aa63b98438de760d6d9b6afb.png)

aided by its widespread adoption and proven performance in language representations. Following the video encoder, we obtain g t from the [CLS] output of BERT, and h t by using the average pooling operation for word-level representations. To match the object and motion representations of videos, here we use the gated embedding unit [11] acting on g t and h t to produce g to , g tm and h to , h tm , respectively.

Audio encoder. Give an audio a, we first extract audio features using a pre-trained VGGish [81], and project these features into the d-dimensional space. As shown in Figure 4, the audio encoder is similar to video encoder in terms of structure. The difference lies in that audio encoder is a single-stream model and has no Cross Encoder. In a similar vein, two different audio representations g a and h a are obtained. The gated embedding unit is also applied to match the object and motion representations of videos.

## B. Anomaly-Led Sampling

As mentioned, only fixed-frame sampling (FS) cannot capture variable anomalous events in anomaly videos. We make use of anomaly priors and propose an anomaly-led sampling (AS) to enable that anomalous clips are more likely to be selected. Since frame-level annotations are unknown, it is impossible to directly identify anomalous clips. To solve this problem, we leverage a weakly supervised anomaly detector to predict clip-level anomaly confidences l ∈ R T , where l i ∈ [0 , 1]. With l in hands, we expect that for a clip, the probability of being selected is positively correlated with its anomaly confidence. A natural way is to select the top several clips with the highest anomaly confidence, but such a solution is too strict to be flexible. We believe that those clips with low anomaly confidences should also have a certain probability to be selected, on the one hand for data augmentation, on the other hand for salvaging false negatives of the anomaly detector. Taking inspiration from the selection strategy of evolutionary algorithms [82], [83], our anomaly-led sampling is based on the classical roulette-wheel selection [84]. To be specific, we regard anomaly confidences l as the fitness, and then normalize all values to the interval [0,1] to ensure summation of selection probabilities equals one,

<!-- formula-not-decoded -->

where p is selection probabilities, and τ is a temperature hyper-parameter [85]. Then calculate cumulative probabilities,

<!-- formula-not-decoded -->

It should be noted that q0 = 0 and qT = 1. The final step, then, is to generate N uniformly distributed random numbers in the interval [0, 1]. For each generated number r, the i-th feature in F(v) is selected if qi − 1 &lt; r ≤ qi
. A sequence with N clip-level features is assembled in such a way, where the larger the anomaly confidence of a clip is, the more likely is its selection. We present the algorithm flow of Anomaly-led sampling in Algorithm 1.

This feature sequence based on anomaly-led sampling are mainly applied to cover anomalous segments, meanwhile, we also use fixed-frame sampling, e.g., uniform or random, to generate another sequence with N clips for the entirety and normal scenarios.

## C. Video Prompt Based Masked Phrase Modeling

We propose a novel pretext task, i.e., video prompt based masked phrase modeling, for cross-modal fine-grained associations in video-text anomaly retrieval. VPMPM takes video representations and text representations as input and predicts the masked phrases, which is related to the prevalent masked

## Algorithm 1: : Anomaly-led sampling based on roulette-wheel selection

Input: anomaly confidence: l; video features: F(v)

Output: N clip-level features step1: Compute the selection probability;

<!-- formula-not-decoded -->

step2: Compute the cumulative probability;

<!-- formula-not-decoded -->

k ← 0;

while k &lt; N do

Step3: Generate a random number r ∈ [0

//

uniform distribution step4: Select features;

if qi

−

1 &lt; r ≤ qi then i-th feature in F(v) is selected;

end k ← k + 1;

end language modeling in nature language processing. The main difference lies in that (1) VPMPM masks and predicts noun phrases and verb phrases instead of randomly selected words. Unlike single words, noun phrases and verb phrases comprise words of different parts of speech, e.g., nouns, adjectives, verbs, adverbs, etc., better correspond to the local objects and motions in video frames; (2) VPMPM fuses video representations with text representations through cross-modal attention, where video representations serve as fixed prompts [23]. Such two specific designs encourage video encoder and text encoder to capture cross-modal and contextual representation interactions.

To achieve this pretext task, we introduce a Prompting Decoder, which is a standard decoder layer used in the Transformer. Since VPMPM involves the objectives of predicting masked noun phrases and masked verb phrases, Prompting Decoder needs to process noun phrases and verb phrases separately in a parameter-shared manner. Given the final video frame-level representations X v and text word-level X t , we first randomly replace a noun phrase or verb phrase representations with mask embeddings [86], where each mask token is a shared, learned vector. Here we denote this masked text representation as X b t . Then we take X b t as the query, and X v as the key and value, feed them into Prompting Decoder to predict the masked contents.

## D. Cross-Modal Alignment

In this paper, cross-modal alignment is used to match representations of different modalities, e.g., video-text and video-audio, from two complementary perspectives. Hence, we deal with CLS alignment and AVG alignment. Unless otherwise stated, here we take video-text as an example to describe these two alignments.

,

1];

CLS alignment. CLS alignment is intended to compute the similarity between g v and g t , and the similarity between them is a weighted sum [13], which is computed as,

<!-- formula-not-decoded -->

where cos(· , · ) is the cosine similarity between two vectors. wta and wtm are weights, which are obtained from g ta and g tm , respectively. Specifically, we pass g ta (g tm ) through a linear layer with softmax normalization, and output wta (wtm). AVG alignment. AVG alignment is intended to compute the similarity s h (v, t) between h v and h t , which is same as CLS alignment. Notably, AVG alignment introduces more finegrained information. The similarity is presented as,

<!-- formula-not-decoded -->

## E. Training Objectives

The final similarity between v and t is the weighted sum of s g (v, t) and s h (v, t), namely,

<!-- formula-not-decoded -->

where α is a hyper-parameter, which lies in the range of [0,1]. Following the previous work [13], we obtain the bi-directional max-margin ranking loss, which is given by,

<!-- formula-not-decoded -->

where B is batch size, sij = s(vi, ti) .

To optimize the weakly supervised anomaly detector in video encoder, we use the top-k strategy [32], [87] to obtain the video-level prediction from frame-level confidences l, which is calculated as,

<!-- formula-not-decoded -->

where k = ⌊ T 16 ⌋, and l topk is the set of k-max framelevel confidences in l for the video v. We train this detector with binary cross-entropy loss Ltopk between the video-level prediction ρ v and video-level binary label y v ,

<!-- formula-not-decoded -->

For VPMPM in video-text anomaly retrieval, we adopt the cross-entropy loss L mpm between the model's predicted probability ρ t (X b t , X v ) and ground truth y mask , which is presented as follows,

<!-- formula-not-decoded -->

where y mask is a one-hot vocabulary distribution.

At last, the overall loss is shown as follows,

<!-- formula-not-decoded -->

TABLE II COMPARISONS WITH THE STATE -OF -THE -ART METHODS ON UCFCRIME-AR.

| Method          | Text→Video    | Text→Video    | Text→Video    | Text→Video    | Video→Text    | Video→Text    | Video→Text    | Video→Text    | SumR↑   |
|-----------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------|
| Method          | R@1↑          | R@5↑          | R@10↑         | MdR↓          | R@1↑          | R@5↑          | R@10↑         | MdR↓          | SumR↑   |
| Random Baseline | 0.3           | 2.1           | 3.4           | 144.0         | 0.3           | 1.0           | 3.1           | 145.5         | 10.2    |
| CE [12]         | 6.6           | 19.7          | 32.4          | 23.5          | 5.5           | 19.7          | 32.4          | 21.0          | 116.3   |
| MMT [13]        | 8.3           | 26.2          | 39.3          | 16.0          | 7.2           | 23.1          | 39.0          | 16.0          | 143.1   |
| T2VLAD [89]     | 7.6           | 23.4          | 39.7          | 15.5          | 6.2           | 27.9          | 43.1          | 14.0          | 147.9   |
| X-CLIP [58]     | 8.2           | 27.2          | 41.7          | 16.0          | 6.9           | 25.8          | 40.3          | 15.0          | 150.1   |
| HL-Net [7]      | 5.5           | 20.2          | 38.3          | 19.5          | 5.5           | 22.8          | 35.5          | 20.0          | 127.8   |
| XML [74]        | 6.9           | 24.1          | 42.4          | 14.0          | 6.6           | 25.9          | 43.4          | 13.0          | 149.3   |
| ALAN            | 9.0           | 27.9          | 44.8          | 14.0          | 7.3           | 24.8          | 46.9          | 12.0          | 160.7   |

TABLE III COMPARISONS WITH THE STATE -OF -THE -ART METHODS ON XDVIOLENCE-AR.

| Method          | Audio→Video    | Audio→Video    | Audio→Video    | Audio→Video    | Video→Audio    | Video→Audio    | Video→Audio    | Video→Audio    | SumR↑   |
|-----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|
| Method          | R@1↑           | R@5↑           | R@10↑          | MdR↓           | R@1↑           | R@5↑           | R@10↑          | MdR↓           | SumR↑   |
| Random Baseline | 0.4            | 0.6            | 2.5            | 399.5          | 0.1            | 0.6            | 0.8            | 399.5          | 5.0     |
| CE [12]         | 11.4           | 33.3           | 47.0           | 12.5           | 13.0           | 34.3           | 46.4           | 13.0           | 185.4   |
| MMT [13]        | 20.5           | 53.5           | 68.0           | 5.0            | 23.0           | 54.6           | 69.5           | 5.0            | 289.1   |
| T2VLAD [89]     | 22.4           | 56.1           | 71.0           | 4.0            | 23.2           | 57.1           | 73.5           | 4.0            | 303.3   |
| X-CLIP [58]     | 26.4           | 61.1           | 73.9           | 3.0            | 26.4           | 61.3           | 73.8           | 4.0            | 322.9   |
| HL-Net [7]      | 12.4           | 36.6           | 48.3           | 11.0           | 13.4           | 38.3           | 52.1           | 10.0           | 201.1   |
| XML [74]        | 22.9           | 55.6           | 70.3           | 5.0            | 22.6           | 57.4           | 71.4           | 4.0            | 300.2   |
| ALAN            | 29.8           | 68.0           | 82.0           | 3.0            | 32.3           | 70.0           | 82.3           | 3.0            | 364.4   |

## V. EXPERIMENTS

## A. Experimental Settings

Evaluation metrics. Following prior works, we use the rankbased metric for performance evaluation, i.e., Recall at K (R@K, K=1, 5, 10), Median Rank (MdR), and Sum of all Recalls (SumR) to measure the overall performance.

Implementation details. We use Spacy 1 to extract noun phrases and verb phrases. In video encoder and audio encoder, the anomaly detector is composed of 3 temporal convolution layers with kernel size of 7, the first layer has 128 units followed by 32 units and 1 unit layers. The first two layers are followed by ReLU, and the last layer is followed by Sigmoid. Dropout with rate of 0.6 is applied to the first two layers. In the text encoder, we use the "BERT-base-cased model" and fine-tune it with a dropout rate of 0.3.

Training. We train our model with a batch size of 64 using Adam [88] optimizer. The initial learning rate is set as 5 × 10 − 5 and decays by a multiplicative factor 0.95 per epoch. For hyper-parameters, hidden size d is set as 768, and temperature parameter τ in Eq. 1 is set as 0.7. Empirically, we found the weight ratio α=0.5 in Eq. 5 and sampling length N=50 worked well across different benchmarks. As the setup in [13], the margin ∆ in Eq. 6 is set as 0.05. λ1 and λ2 in Eq. 10 is set as 0.1 and 0.01, respectively, such a setup achieves optimal performance.

## B. Comparison with State-of-the-Art Methods

We conduct experiments on UCFCrime-AR and XDViolence-AR and compare our ALAN with several recent methods that are widely used in video retrieval, video moment retrieval and VAD. CE [12], MMT [13],

T2VLAD [89], and X-CLIP [58] are video retrieval methods; XML [74] is a video moment retrieval method, here it is used to retrieve videos, where the moment localization part is removed since moment annotations are unavailable in VAR. HL-Net [7] is a VAD method, since VAD is quite distinct from VAR, it is hard to directly use VAD method for VAR, here, we modify it as a video encoder for VAR. All methods use BERT to extract language features except CE that uses the word2vec word embeddings [90]. We present comparison results in Tables II and III, and observe that our ALAN shows a clear advantage over comparison methods in both text-video and audio-video VAR. Specifically, ALAN outperforms CE, MMT, T2VLAD, X-CLIP, HL-Net, and XML on UCFCrime-AR by 44.4, 17.6, 12.8, 10.6, 32.9, and 11.4 in terms of SumR, respectively. Furthermore, ALAN also achieves clear improvements against competitors on XDViolence-AR, which achieves a significant performance improvement of 41.5 in terms of SumR over the previous best method. Moreover, It can be found that, in comparison to the video and text, the video and audio are easier to align. We argue that video and audio are synchronous with concordant granularity, thereby leading to better align performance in VAR.

## C. Ablation Studies

Study on anomaly-led sampling. As aforementioned, we propose a novel sampling mechanism, i.e., anomaly-led sampling, which combines with the ordinary fixed-frame sampling, and the joint effort is devoted to capturing local anomalous segments as well as overall information. To investigate the effectiveness of anomaly-led sampling, we conduct experiments on two benchmarks, and show results on Tables IV and V. As we can see from the first two rows, only using fixed-frame sampling or anomaly-led sampling results in a clear performance

TABLE IV COMPARISONS OF DIFFERENT SAMPLINGS ON UCFCRIME-AR.

| Sampling     | Text→Video    | Text→Video    | Video→Text   | Video→Text   |
|--------------|---------------|---------------|--------------|--------------|
| Sampling     | R@1↑          | R@10↑         | R@1↑         | R@10↑        |
| FS (N=50)    | 6.6           | 35.5          | 4.8          | 42.4         |
| AS (N=50)    | 7.9           | 37.6          | 5.5          | 41.7         |
| FS (N=100)   | 6.6           | 37.6          | 6.2          | 40.3         |
| FS+AS (N=50) | 9.0           | 44.8          | 7.3          | 46.9         |

TABLE V COMPARISONS OF DIFFERENT SAMPLINGS ON XDVIOLENCE-AR.

| Sampling     | Text→Video    | Text→Video    | Video→Text   | Video→Text   |
|--------------|---------------|---------------|--------------|--------------|
| Sampling     | R@1↑          | R@10↑         | R@1↑         | R@10↑        |
| FS (N=50)    | 29.6          | 80.4          | 31.1         | 80.9         |
| AS (N=50)    | 26.9          | 78.6          | 27.4         | 78.9         |
| FS (N=100)   | 28.5          | 81.0          | 29.8         | 81.8         |
| FS+AS (N=50) | 29.8          | 82.0          | 32.3         | 82.3         |

drop on both UCFCrime-AR and XDViolence-AR. Besides, using anomaly-led sampling is inferior to using fixed-sampling on XDViolence-AR, we discover that the main reason for this problem is that the anomaly-led sampling mechanism is applied to both video and audio, resulting in key segments misalignment to some extent. Moreover, we also investigate the effect of sampling length. From the third row, we found that increasing the sampling length from 50 to 100 does not dramatically improve performance, and fixed-frame sampling still lags behind the combination of fixed-frame sampling and anomaly-led sampling, even though they both have the same sampling length at the moment. It also clearly demonstrates that the joint effect between anomaly-led sampling and fixedframe sampling enables our model to capture key anomalous segments as well as holistic data information, thus facilitating cross-modal alignment under local-anomaly and global-video perspectives. For example, in Figure 8, video frames that are selected by anomaly-led sampling are aligned with the key anomaly descriptions, e.g., two car collided violently, a man in black lay on the ground and shot. On another hand, these video frames selected by fixed-frame sampling are aligned with the complete descriptions.

Study on VPMPM. Here we conduct experiments to certify the advantage of VPMPM for video-text fine-grained associations. When ALAN removes VPMPM at training time, we observe the performance clearly drops as shown in Table VI. Besides, masking and predicting in the form of random words rather than noun phrases and verb phrases in VPMPM hurts performance. We can also see that, using noun phrases and verb phrases are superior to noun and verb words on most evaluation metrics. This demonstrates that noun phrases and verb phrases, as the sequences of words with different parts of speech, can better align with related local contents in videos. Study on cross-modal alignment. Tables VII and VIII present the performance of two different alignments in our ALAN. We found that CLS alignment and AVG alignment obtain worse results when used alone in comparison to the model of jointly using both. Such results demonstrate the complementarity of these two alignments. A key observation is the AVG alignment performs better than CLS alignment on XDViolence-AR, but

TABLE VI VPMPM STUDIES ON UCFCRIME-AR.

| Method            | Text→Video    | Text→Video    | Video→Text   | Video→Text   |
|-------------------|---------------|---------------|--------------|--------------|
| Method            | R@1↑          | R@10↑         | R@1↑         | R@10↑        |
| w/o VPMPM         | 7.9           | 43.4          | 7.2          | 43.4         |
| random words      | 8.6           | 43.8          | 6.2          | 44.5         |
| noun&verb words   | 10.0          | 44.5          | 6.6          | 42.8         |
| noun&verb phrases | 9.0           | 44.8          | 7.3          | 46.9         |

TABLE VII COMPARISONS OF DIFFERENT ALIGNMENTS ON UCFCRIME-AR.

TABLE VIII COMPARISONS OF DIFFERENT ALIGNMENTS ON XDVIOLENCE-AR.

| Alignment    | Text→Video    | Text→Video    | Video→Text   | Video→Text   |
|--------------|---------------|---------------|--------------|--------------|
| Alignment    | R@1↑          | R@10↑         | R@1↑         | R@10↑        |
| CLS          | 6.2           | 42.8          | 7.6          | 40.3         |
| AVG          | 6.6           | 33.8          | 4.8          | 36.9         |
| CLS+AVG      | 9.0           | 44.8          | 7.3          | 46.9         |

Fig. 5. Influences of α on both UCFCrime-AR and XDViolence-AR.

| Alignment    | Audio→Video    | Audio→Video    | Video→Audio   | Video→Audio   |
|--------------|----------------|----------------|---------------|---------------|
| Alignment    | R@1↑           | R@10↑          | R@1↑          | R@10↑         |
| CLS          | 26.8           | 77.9           | 28.6          | 77.4          |
| AVG          | 28.0           | 79.3           | 30.0          | 80.1          |
| CLS+AVG      | 29.8           | 82.0           | 32.3          | 82.3          |

![Image](artifacts/image_000004_1fa8a41fdc19f493ea799cd30b866cdb173f596f5749198cec0a759a937b2a16.png)

the opposite is true on UCFCrime-AR, we suspect that video and audio are easier to align at the fine-grained level due to their concordant granularity. Moreover, we also investigate the influence of α. We try α with its value ranging from 0.0 to 1.0 with an interval of 0.1. As shown in Figure 5, with the increase of α, the performance gradually improves and then decreases, when α is set as 0.5, our method achieves the best performance. In order to further explore how to choose α , we also show the detailed retrieval results of different α in Tables IX and X. It is not hard to see that it is a balanced choice to set the value range of α to 0.4-0.6, where two different cross-modal alignments make nearly the same contribution.

## D. Qualitative Analyses

Visualization of retrieval results. Some text-to-video retrieval examples on UCFCrime-AR are exhibited in Figure 6, where retrieval results of a normal video is shown at the far right. We observe ALAN successfully retrieves the related video given

Fig. 6. Some retrieval examples on UCFCrime-AR. We visualize top 3 retrieved videos (green: correct; pink: incorrect).

![Image](artifacts/image_000005_50065715b9d076fcfdf43b5d22aaefcb3d327e2b96c3249fb73a1e9e52c04f76.png)

TABLE IX DETAILED INFLUENCES OF α ON UCFCRIME-AR.

| Value of α   | Audio→Video    | Audio→Video    | Video→Audio   | Video→Audio   |
|--------------|----------------|----------------|---------------|---------------|
| Value of α   | R@1↑           | R@10↑          | R@1↑          | R@10↑         |
| 0.0          | 6.6            | 33.8           | 4.8           | 36.9          |
| 0.2          | 8.3            | 38.6           | 6.2           | 40.3          |
| 0.4          | 7.9            | 42.4           | 6.9           | 44.5          |
| 0.5          | 9.0            | 44.8           | 7.3           | 46.9          |
| 0.6          | 7.6            | 45.2           | 6.2           | 47.2          |
| 0.8          | 7.6            | 43.8           | 5.5           | 43.4          |
| 1.0          | 6.2            | 42.8           | 7.6           | 40.3          |

TABLE X DETAILED INFLUENCES OF α ON UCFCRIME-AR.

| Value of α   | Audio→Video    | Audio→Video    | Video→Audio   | Video→Audio   |
|--------------|----------------|----------------|---------------|---------------|
| Value of α   | R@1↑           | R@10↑          | R@1↑          | R@10↑         |
| 0.0          | 28.0           | 79.3           | 30.0          | 80.1          |
| 0.2          | 30.4           | 80.8           | 32.1          | 82.6          |
| 0.4          | 31.9           | 80.9           | 32.3          | 82.4          |
| 0.5          | 29.8           | 82.0           | 32.3          | 82.3          |
| 0.6          | 31.0           | 82.6           | 32.8          | 80.8          |
| 0.8          | 28.0           | 79.8           | 30.1          | 79.3          |
| 1.0          | 26.8           | 77.9           | 28.6          | 77.4          |

a text query, and there are considerable similarities between the top 3 retrieved videos. This also demonstrates VAR is a challenging task as some scenes are similar with delicate differences.

Visualization of coarse caption retrieval. In VAR task, the purpose of using accurate captions is to distinguish fine differences and avoid being into a one-to-many dilemma [69]. To further verify the generalization capacity of ALAN, we use several coarse captions that are not directly applied in model training to retrieve videos, results in Figure 7 clearly show that ALAN works very well with different lengths of coarse captions, and also demonstrate ALAN has learned several abstract semantic information, e.g., explosion, fighting, traffic. This also convincingly indicates our methods can meet practical requirements where users cannot provide a complete text description of the videos they intend to search, such as the example in the lower right of Figure 7, users give the the retrieval model a incomplete description "man robbed people", and the model returns top 3 related videos, in which the contents correspond to robbery, steal, man, and people.

Visualization of anomaly-led sampling. We visualize video frames selected by fixed-frame sampling and anomaly-led sampling in Figure 8. These examples are taken from videos

Fig. 7. Some coarse caption retrieval examples on UCFCrime-AR.

![Image](artifacts/image_000006_8cbe8b72a6adfb6c8591506aa5ac370cd4b1e7fdf58c7a2c1ea47d26487ee223.png)

of road accident and shooting scenes. It can be seen from the second row that the duration of anomalous event accounts for less than one-fifth of the entire video length, therefore, frames related to the anomalous event are hard to select based on fixed-frame sampling. In stark contrast to fixedframe sampling, anomaly-led sampling is based on anomaly confidences generated by the anomaly detector, and it can select more frames related to anomalous events since the probability of being selected has positive correlations with anomaly confidences, where anomaly detector generates high confidences in anomalous segments which is shown in the second row.

Visualization of zero-shot retrieval. ALAN is trained on UCFCrime-AR and XDViolence-AR for text-video and audiovideo anomaly retrieval, respectively. Moreover, scenarios in these two benchmarks are different, because videos from UCFCrime-AR are captured with fixed cameras, whereas videos from XDViolence-AR are collected from movies and YouTube. Here we explore that, given a cross-modal query from UCFCrime-AR (or XDViolence-AR), is ALAN trained on UCFCrime-AR (or XDViolece-AR) capable of retrieving some relevant videos from XDViolence-AR (or texts from UCFCrime-AR)? We show the top 2 retrieval results in Figure 9. In text-to-video anomaly retrieval, we found that given text queries from UCFCrime-AR, ALAN can retrieve some videos from XDViolence-AR that look semantically plausible, even if there are no completely relevant videos in XDViolence-AR. Interestingly, the video in the bottom

Fixed

-

Frame

Sampling

Anomaly

Confidence

Anomaly-Led

Sampling

GT

At night

,

the two cars collided violently in the middle of the crossroad and crashed into the side of the road

.

Time Axis

![Image](artifacts/image_000007_22cc091177052528049679d9da8334a2a6ff611ee5116f01e6a2b7e84fb76d83.png)

![Image](artifacts/image_000008_e57f5efa0d6f55780c2fc18ca441e75c25108b1723deb362256ea921a782f9ea.png)

Fig. 8. Different samplings for video frame selection. Left: road accident; Right: Shooting.

![Image](artifacts/image_000009_169a0f3bf67607415c77ee28a05f4745173aea22994081870a4f65a2c5f55d82.png)

Text

-

to-Video

Video

-

to-Text

Fig. 9. Zero-shot retrieval results. The left two columns present zero-shot text-to-video anomaly retrieval, and the right two columns present zero-shot video-to-text anomaly retrieval.

left is an animation. ALAN learns several local semantic contents and retrieves videos based on these local semantic contents, such as "huge fire" and "mushroom cloud". In videoto-text anomaly retrieval, although retrieved text descriptions are not completely related, ALAN captures partial semantic information from movie videos, such as "a man", "a female companion", "knock down somebody with fists", etc.

## E. Running Time

We report the retrieval time for UCFCrime-AR with 290 video-text test pairs and XDViolence-AR with 800 videoaudio test pairs, our method costs 2.7s and 5.6s, respectively. Generally, it only needs about 0.008s to process a pair on both datasets, showing its higher efficiency. The reason why our method remains high retrieval efficiency is that it has a dual-encoder structure during the test stage, that is, using two separate encoders to embed video and text features and project them into the latent joint space, and only the cosine similarity between video and text features is calculated as similarity, without complicated and inefficient cross-modal interactions. However, it is worth noting that, during the training phase, our method integrates text and video as inputs to a joint encoder for the cross-modality fusion, which can establish local correlation between video-text features and improve retrieval accuracy. Therefore, our method obtains the advantages of the above two kinds of methods, that is, achieving finegrained video-text interactions while maintaining high retrieval efficiency.

## VI. CONCLUSION

In this paper, we introduce a new task called video anomaly retrieval to remedy the inadequacy of video anomaly de- tection in terms of abnormal event depict, further facilitate video anomaly analysis research in cross-modal scenarios. We construct two VAR benchmarks, i.e., UCFCrime-AR and XDViolence-AR, based on popular VAD datasets. Moreover, we propose ALAN which includes several components, where anomaly-led sampling is used to capture local anomalous segments, which coordinates with ordinary fixed-frame sampling to achieve complementary effects; Video prompt based masked phrase modeling is used to learn cross-modal finegrained associations; Cross-modal alignment is used to match cross-modal representations from two perspectives. The future work will lie in two aspects, 1) exploiting cross-modal pretrained models to capture more powerful knowledge for VAR; 2) leveraging VAR to assist VAD methods for more precise anomaly detection.

## REFERENCES

- [1] M. Sabokrou, M. Fayyaz, M. Fathy, and R. Klette, "Deep-cascade: Cascading 3d deep neural networks for fast anomaly detection and localization in crowded scenes," IEEE Transactions on Image Processing , vol. 26, no. 4, pp. 1992–2004, 2017.
- [2] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly detection–a new baseline," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6536–6545.
- [3] P. Wu, J. Liu, and F. Shen, "A deep one-class neural network for anomalous event detection in complex scenes," IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 7, pp. 2609–2622, 2019.
- [4] H. Park, J. Noh, and B. Ham, "Learning memory-guided normality for anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 14 372–14 381.
- [5] M.-I. Georgescu, A. Barbalau, R. T. Ionescu, F. S. Khan, M. Popescu, and M. Shah, "Anomaly detection in video via self-supervised and multi-task learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12 742–12 752.

- [6] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 6479–6488.
- [7] P. Wu, J. Liu, Y. Shi, Y. Sun, F. Shao, Z. Wu, and Z. Yang, "Not only look, but also listen: Learning multimodal violence detection under weak supervision," in Proceedings of the European Conference on Computer Vision. Springer, 2020, pp. 322–339.
- [8] J.-C. Feng, F.-T. Hong, and W.-S. Zheng, "Mist: Multiple instance selftraining framework for video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 14 009–14 018.
- [9] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro, "Weakly-supervised video anomaly detection with robust temporal feature magnitude learning," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.
- [10] J. Wu, W. Zhang, G. Li, W. Wu, X. Tan, Y. Li, E. Ding, and L. Lin, "Weakly-supervised spatio-temporal anomaly detection in surveillance video," arXiv preprint arXiv:2108.03825, 2021.
- [11] A. Miech, I. Laptev, and J. Sivic, "Learning a text-video embedding from incomplete and heterogeneous data," arXiv preprint arXiv:1804.02516 , 2018.
- [12] Y. Liu, S. Albanie, A. Nagrani, and A. Zisserman, "Use what you have: Video retrieval using representations from collaborative experts," arXiv preprint arXiv:1907.13487, 2019.
- [13] V. Gabeur, C. Sun, K. Alahari, and C. Schmid, "Multi-modal transformer for video retrieval," in Proceedings of the 16th European Conference on Computer Vision. Springer, 2020, pp. 214–229.
- [14] X. Yang, S. Wang, J. Dong, J. Dong, M. Wang, and T.-S. Chua, "Video moment retrieval with cross-modal neural architecture search," IEEE Transactions on Image Processing, vol. 31, pp. 1204–1216, 2022.
- [15] A. Yang, A. Miech, J. Sivic, I. Laptev, and C. Schmid, "Tubedetr: Spatiotemporal video grounding with transformers," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 16 442–16 453.
- [16] R. Cui, T. Qian, P. Peng, E. Daskalaki, J. Chen, X. Guo, H. Sun, and Y.-G. Jiang, "Video moment retrieval from text queries via single frame annotation," in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval , 2022, pp. 1033–1043.
- [17] G. Wang, X. Xu, F. Shen, H. Lu, Y. Ji, and H. T. Shen, "Cross-modal dynamic networks for video moment retrieval with text query," IEEE Transactions on Multimedia, vol. 24, pp. 1221–1232, 2022.
- [18] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, "Dynamic neural networks: A survey," arXiv preprint arXiv:2102.04906, 2021.
- [19] Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, "Dynamicvit: Efficient vision transformers with dynamic token sparsification," arXiv preprint arXiv:2106.02034, 2021.
- [20] Y. Zhi, Z. Tong, L. Wang, and G. Wu, "Mgsampler: An explainable sampling strategy for video action recognition," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.
- [21] M. Fayyaz, S. A. Koohpayegani, F. R. Jafari, S. Sengupta, H. R. V. Joze, E. Sommerlade, H. Pirsiavash, and J. Gall, "Adaptive token sampling for efficient vision transformers," in Proceedings of the European Conference on Computer Vision. Springer, 2022, pp. 396–414.
- [22] J. Li, R. Selvaraju, A. Gotmare, S. Joty, C. Xiong, and S. C. H. Hoi, "Align before fuse: Vision and language representation learning with momentum distillation," Advances in Neural Information Processing Systems, vol. 34, pp. 9694–9705, 2021.
- [23] Z. Hou, F. Sun, Y.-K. Chen, Y. Xie, and S.-Y. Kung, "Milan: Masked image pretraining on language assisted representation," arXiv preprint arXiv:2208.06049, 2022.
- [24] S. Lee, H. G. Kim, and Y. M. Ro, "Bman: Bidirectional multi-scale aggregation networks for abnormal event detection," IEEE Transactions on Image Processing, vol. 29, pp. 2395–2408, 2019.
- [25] R. T. Ionescu, F. S. Khan, M.-I. Georgescu, and L. Shao, "Object-centric auto-encoders and dummy anomalies for abnormal event detection in video," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 7842–7851.
- [26] D. Gong, L. Liu, V. Le, B. Saha, M. R. Mansour, S. Venkatesh, and A. v. d. Hengel, "Memorizing normality to detect anomaly: Memoryaugmented deep autoencoder for unsupervised anomaly detection," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1705–1714.
- [27] G. Wang, Y. Wang, J. Qin, D. Zhang, X. Bao, and D. Huang, "Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles," 2022.
- [28] Z. Yang, P. Wu, J. Liu, and X. Liu, "Dynamic local aggregation network with adaptive clusterer for anomaly detection," in Proceedings of the European Conference on Computer Vision. Springer, 2022, pp. 404– 421.
- [29] Z. Yang, J. Liu, Z. Wu, P. Wu, and X. Liu, "Video event restoration based on keyframes for video anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 14 592–14 601 .
- [30] C. Yan, S. Zhang, Y. Liu, G. Pang, and W. Wang, "Feature prediction diffusion model for video anomaly detection," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5527–5537 .
- [31] H. Lv, C. Zhou, Z. Cui, C. Xu, Y. Li, and J. Yang, "Localizing anomalies from weakly-labeled videos," IEEE transactions on image processing , vol. 30, pp. 4505–4515, 2021.
- [32] P. Wu and J. Liu, "Learning causal temporal relation and feature discrimination for anomaly detection," IEEE Transactions on Image Processing, vol. 30, pp. 3513–3527, 2021.
- [33] C. Cao, X. Zhang, S. Zhang, P. Wang, and Y. Zhang, "Adaptive graph convolutional networks for weakly supervised anomaly detection in videos," arXiv preprint arXiv:2202.06503, 2022.
- [34] C. Huang, C. Liu, J. Wen, L. Wu, Y. Xu, Q. Jiang, and Y. Wang, "Weakly supervised video anomaly detection via self-guided temporal discriminative transformer," IEEE Transactions on Cybernetics, 2022 .
- [35] Y. Peng, X. Huang, and Y. Zhao, "An overview of cross-media retrieval: Concepts, methodologies, benchmarks, and challenges," IEEE Transactions on Circuits and Systems for Video Technology, vol. 28, no. 9, pp. 2372–2385, 2017.
- [36] Y. Peng, W. Zhu, Y. Zhao, C. Xu, Q. Huang, H. Lu, Q. Zheng, T. Huang, and W. Gao, "Cross-media analysis and reasoning: advances and directions," Frontiers of Information Technology &amp; Electronic Engineering , vol. 18, no. 1, pp. 44–57, 2017.
- [37] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu, "Coca: Contrastive captioners are image-text foundation models," arXiv preprint arXiv:2205.01917, 2022.
- [38] R. Zuo, X. Deng, K. Chen, Z. Zhang, Y.-K. Lai, F. Liu, C. Ma, H. Wang, Y.-J. Liu, and H. Wang, "Fine-grained video retrieval with scene sketches," IEEE Transactions on Image Processing, 2023.
- [39] M. Monfort, S. Jin, A. Liu, D. Harwath, R. Feris, J. Glass, and A. Oliva, "Spoken moments: Learning joint audio-visual representations from video descriptions," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 14 871–14 881.
- [40] A.-M. Oncescu, A. Koepke, J. F. Henriques, Z. Akata, and S. Albanie, "Audio retrieval with natural language queries," arXiv preprint arXiv:2105.02192, 2021.
- [41] V. Gabeur, A. Nagrani, C. Sun, K. Alahari, and C. Schmid, "Masking modalities for cross-modal video retrieval," arXiv preprint arXiv:2111.01300, 2021.
- [42] A. Rouditchenko, A. Boggust, D. Harwath, S. Thomas, H. Kuehne, B. Chen, R. Panda, R. Feris, B. Kingsbury, M. Picheny et al., "Cascaded multilingual audio-visual learning from videos," arXiv preprint arXiv:2111.04823, 2021.
- [43] P. Morgado, N. Vasconcelos, and I. Misra, "Audio-visual instance discrimination with cross-modal agreement," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 12 475–12 486.
- [44] W. Shen, J. Song, X. Zhu, G. Li, and H. T. Shen, "End-to-end pretraining with hierarchical matching and momentum contrast for textvideo retrieval," IEEE Transactions on Image Processing, 2023.
- [45] Y. Tian, J. Shi, B. Li, Z. Duan, and C. Xu, "Audio-visual event localization in unconstrained videos," in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 247–263 .
- [46] Y. Wei, D. Hu, Y. Tian, and X. Li, "Learning in audio-visual context: A review, analysis, and new perspective," arXiv preprint arXiv:2208.09579 , 2022 .
- [47] Y. Wu, L. Zhu, Y. Yan, and Y. Yang, "Dual attention matching for audiovisual event localization," in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 6292–6300 .
- [48] Y.-B. Lin, Y.-L. Sung, J. Lei, M. Bansal, and G. Bertasius, "Vision transformers are parameter-efficient audio-visual learners," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 2299–2309 .
- [49] X. Li, C. Xu, G. Yang, Z. Chen, and J. Dong, "W2vv++ fully deep learning for ad-hoc video search," in Proceedings of the 27th ACM International Conference on Multimedia, 2019, pp. 1786–1794.

- [50] J. Dong, X. Li, C. Xu, X. Yang, G. Yang, X. Wang, and M. Wang, "Dual encoding for video retrieval by text," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
- [51] S. Liu, H. Fan, S. Qian, Y. Chen, W. Ding, and Z. Wang, "Hit: Hierarchical transformer with momentum contrast for video-text retrieval," arXiv preprint arXiv:2103.15049, 2021.
- [52] M. Wray, D. Larlus, G. Csurka, and D. Damen, "Fine-grained action retrieval through multiple parts-of-speech embeddings," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 450–459.
- [53] P. Wu, X. He, M. Tang, Y. Lv, and J. Liu, "Hanet: Hierarchical alignment networks for video-text retrieval," in Proceedings of the 29th ACM International Conference on Multimedia, 2021, pp. 3518–3527.
- [54] N. Han, J. Chen, G. Xiao, H. Zhang, Y. Zeng, and H. Chen, "Finegrained cross-modal alignment network for text-video retrieval," in Proceedings of the 29th ACM International Conference on Multimedia , 2021, pp. 3826–3834.
- [55] J. Yang, Y. Bisk, and J. Gao, "Taco: Token-aware cascade contrastive learning for video-text alignment," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 11 562–11 572.
- [56] W. Wang, M. Zhang, R. Chen, G. Cai, P. Zhou, P. Peng, X. Guo, J. Wu, and X. Sun, "Dig into multi-modal cues for video retrieval with hierarchical alignment," in Proceedings of the International Joint Conference on Artificial Intelligence, 2021.
- [57] Y. Ge, Y. Ge, X. Liu, D. Li, Y. Shan, X. Qie, and P. Luo, "Bridging video-text retrieval with multiple choice questions," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 16 167–16 176.
- [58] Y. Ma, G. Xu, X. Sun, M. Yan, J. Zhang, and R. Ji, "X-clip: Endto-end multi-grained contrastive learning for video-text retrieval," in Proceedings of the 30th ACM International Conference on Multimedia , 2022, pp. 638–647.
- [59] L. Li, Y.-C. Chen, Y. Cheng, Z. Gan, L. Yu, and J. Liu, "Hero: Hierarchical encoder for video+ language omni-representation pre-training," arXiv preprint arXiv:2005.00200, 2020.
- [60] J. Lei, L. Li, L. Zhou, Z. Gan, T. L. Berg, M. Bansal, and J. Liu, "Less is more: Clipbert for video-and-language learning via sparse sampling," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 7331–7341.
- [61] H. Luo, L. Ji, B. Shi, H. Huang, N. Duan, T. Li, J. Li, T. Bharti, and M. Zhou, "Univl: A unified video and language pre-training model for multimodal understanding and generation," arXiv preprint arXiv:2002.06353, 2020.
- [62] K. Ji, J. Liu, W. Hong, L. Zhong, J. Wang, J. Chen, and W. Chu, "Cret: Cross-modal retrieval transformer for efficient text-video retrieval," in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 949– 959.
- [63] J. Dong, X. Chen, M. Zhang, X. Yang, S. Chen, X. Li, and X. Wang, "Partially relevant video retrieval," in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 246–257.
- [64] J. Gao, C. Sun, Z. Yang, and R. Nevatia, "Tall: Temporal activity localization via language query," in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 5267–5275.
- [65] D. Zhang, X. Dai, X. Wang, Y.-F. Wang, and L. S. Davis, "Man: Moment alignment network for natural language moment retrieval via iterative graph adjustment," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 1247–1257.
- [66] N. C. Mithun, S. Paul, and A. K. Roy-Chowdhury, "Weakly supervised video moment retrieval from text queries," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019, pp. 11 592–11 601.
- [67] X. Ding, N. Wang, S. Zhang, Z. Huang, X. Li, M. Tang, T. Liu, and X. Gao, "Exploring language hierarchy for video grounding," IEEE Transactions on Image Processing, vol. 31, pp. 4693–4706, 2022.
- [68] A. Miech, D. Zhukov, J.-B. Alayrac, M. Tapaswi, I. Laptev, and J. Sivic, "Howto100m: Learning a text-video embedding by watching hundred million narrated video clips," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 2630–2640.
- [69] M. Wray, H. Doughty, and D. Damen, "On semantic similarity in video retrieval," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3650–3660.
- [70] J. Xu, T. Mei, T. Yao, and Y. Rui, "Msr-vtt: A large video description dataset for bridging video and language," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2016, pp. 5288–5296.
- [71] X. Wang, J. Wu, J. Chen, L. Li, Y.-F. Wang, and W. Y. Wang, "Vatex: A large-scale, high-quality multilingual dataset for video-and-language research," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 4581–4591.
- [72] C. D. Kim, B. Kim, H. Lee, and G. Kim, "Audiocaps: Generating captions for audios in the wild," in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019, pp. 119–132.
- [73] Y. Tian, D. Li, and C. Xu, "Unified multisensory perception: Weaklysupervised audio-visual video parsing," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, 2020, pp. 436–454 .
- [74] J. Lei, L. Yu, T. L. Berg, and M. Bansal, "Tvr: A large-scale dataset for video-subtitle moment retrieval," in Proceedings of the 16th European Conference on Computer Vision. Springer, 2020, pp. 447–463.
- [75] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Luciˇ ˇ c, and C. Schmid, ´ ´ "Vivit: A video vision transformer," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6836–6846.
- [76] D. Neimark, O. Bar, M. Zohar, and D. Asselmann, "Video transformer network," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 3163–3172.
- [77] J. Carreira and A. Zisserman, "Quo vadis, action recognition? a new model and the kinetics dataset," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.
- [78] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in neural information processing systems, 2017, pp. 5998–6008.
- [79] C. Huang, Y. Liu, Z. Zhang, C. Liu, J. Wen, Y. Xu, and Y. Wang, "Hierarchical graph embedded pose regularity learning via spatio-temporal transformer for abnormal behavior detection," in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 307–315 .
- [80] S. Zhao, L. Zhu, X. Wang, and Y. Yang, "Centerclip: Token clustering for efficient text-video retrieval," in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 970–981 .
- [81] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, "Audio set: An ontology and humanlabeled dataset for audio events," in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2017, pp. 776–780.
- [82] K. Wu, J. Liu, X. Hao, P. Liu, and F. Shen, "An evolutionary multiobjective framework for complex network reconstruction using community structure," IEEE Transactions on Evolutionary Computation, vol. 25, no. 2, pp. 247–261, 2020.
- [83] Y. Jin, H. Wang, T. Chugh, D. Guo, and K. Miettinen, "Data-driven evolutionary optimization: An overview and case studies," IEEE Transactions on Evolutionary Computation, vol. 23, no. 3, pp. 442–458, 2018.
- [84] T. Back, Evolutionary algorithms in theory and practice: evolution strategies, evolutionary programming, genetic algorithms. Oxford university press, 1996.
- [85] Z. Wu, Y. Xiong, S. X. Yu, and D. Lin, "Unsupervised feature learning via non-parametric instance discrimination," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3733–3742.
- [86] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, "Masked au- ´ ´ toencoders are scalable vision learners," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 000–16 009.
- [87] S. Paul, S. Roy, and A. K. Roy-Chowdhury, "W-talc: Weakly-supervised temporal activity localization and classification," in Proceedings of the European Conference on Computer Vision, 2018, pp. 563–579.
- [88] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.
- [89] X. Wang, L. Zhu, and Y. Yang, "T2vlad: global-local sequence alignment for text-video retrieval," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 5079–5088.
- [90] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," arXiv preprint arXiv:1301.3781 , 2013.