---
layout: default
title: Home
---
## About Me

I am an MRes (Master of Research) student in Machine Learning at Imperial College London, supervised by [**Dr. Matthieu Komorowski**](https://www.linkedin.com/in/matthieukomorowski/) and [**Dr. Guang Yang**](https://www.linkedin.com/in/gyangmedia/). I obtained my B.Sci. degree in Statistic with a minor in Computing from The University of Melbourne and an Honours degree in Data Science from The University of Sydney.


- My current research focuses on applying advanced AI techniques, including:
  - **GANs** and **Diffusion Generative models**, to **medical imaging**.
  - **Medical data Synthesis** 
  -  **CXR image translation and Computer-aid Medical Image Analysis**.

**I am actively seeking PhD and RA opportunities.**

Feel free to contact me for Chat and Research Opportunities. 


_"Positivity is the essence of progress. In every challenge, I see an opportunity for learning and growth."_


## Education


<!-- ## custom laying the image in the html file -->
<!-- <img src="/junzhin-homepage/assets/img/header_icon.png" alt="Header Icon" style="float: left; margin-right: 20px;"> -->

<!-- ## default way of rending the image in the html through markdown sytnax -->

<!-- ![Header Icon](/assets/img/header_icon.png) -->

- **Master of Research (MRes)**, Imperial College London, UK _(Sep 2023 - Oct 2024)_  (Grade: Distinction - First-Class Honours)
  - Major: Artificial Intelligence and Machine Learning 
  - Research supervised by **Dr. Guang Yang** and **Dr. Matthieu Komorowski**, focusing on medical imaging and AI.

- **Bachelor of Science (Honours)**, The University of Sydney, Australia _(Aug 2022 - Aug 2023)_ (Overall WAM 89.5 with university medal: First-Class Honours)
  - Major: Data Science . 
  - Achieved First Class Honours with a **University Medal**.
  - Thesis supervised by **Dr. Liu Tongliang** and **Dr. Mingming Gong** jointly.

- **Bachelor of Science**, The University of Melbourne, Australia _(Feb 2019 - Jul 2022)_ (Overall WAM: 86.8/First-Class Honours)
  - Major: Statistics and Computing 
  - Achieved First Class Honours.

## Employment and Research Experience

- (Nov 2024 - Present) **Research Assistant** – Shanghai AI Lab, Shanghai, China
- (Oct 2023 - Oct 2024) **Research Graduate Assistant** – Imperial College London, London, UK
- (Aug 2022 - Jun 2023) **Research Undergraduate Assistant** – Trustworthy Machine Learning Lab, University of Sydney, Sydney, Australia
- (Nov 2022 - Jan 2023) **Software Engineering Intern** – Guangzhou Junrui Kang Biotechnology, Guangzhou, China
- (Dec 2021 - Jun 2022) **Research Undergraduate Assistant** – University of Melbourne, Melbourne, Australia




## Selected Publication and Research Work

- **GMAI-VL-R1: Harnessing Reinforcement Learning for Multimodal Medical Reasoning**  
  _NeurIPS 2025, Co-author, Under Review_
  
  <div style="overflow: hidden;">
   <img src="/junzhin-homepage/assets/img/GMAI_R1.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
    <div class="text-justified">
      Recent advances in general medical AI have made significant strides, but existing models often lack the reasoning capabilities needed for complex medical decision-making. This paper presents GMAI-VL-R1, a multimodal medical reasoning model enhanced by reinforcement learning (RL) to improve its reasoning abilities. Through iterative training, GMAI-VL-R1 optimizes decision-making, significantly boosting diagnostic accuracy and clinical support. We also develop a reasoning data synthesis method, generating step-by-step reasoning data via rejection sampling, which further enhances the model's generalization. Experimental results show that after RL training, GMAI-VL-R1 excels in tasks such as medical image diagnosis and visual question answering. While the model demonstrates basic memorization with supervised fine-tuning, RL is crucial for true generalization. Our work establishes new evaluation benchmarks and paves the way for future advancements in medical reasoning models. Code, data, and model will be released.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **RetinaLogos: Fine-Grained Synthesis of High-Resolution Retinal Images Through Captions**  
  _MICCAI 2025, **First author**, Rebuttal_ [Code](https://anonymous.4open.science/r/Text-Driven-CFP-Generator)
  
  <div style="overflow: hidden;">
  
    <img src="/junzhin-homepage/assets/img/retina_logos.png" alt="RetinaLogos" style="float: left; margin-right: 20px; width: 150px;">
    
    <div class="text-justified">
      The scarcity of high-quality, labelled retinal imaging data presents a significant challenge in the development of machine learning models for ophthalmology. To address this, we introduce RetinaLogos-1400k, a large-scale synthetic Caption-CFP dataset with 1.4 million entries that uses large language models to describe retinal conditions and key structures. Based on this dataset, we employ a novel three-step training framework that enables fine-grained semantic control over retinal images and accurately captures different stages of disease progression and anatomical variations. Extensive experiments show that 62.07% of our text-driven synthetic images are indistinguishable from real ones by ophthalmologists, and the synthetic data improves accuracy by 10%-25% in diabetic retinopathy grading and glaucoma detection.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model**  
  _MICCAI 2025, Co-author, Early Accepted_
  
  <div style="overflow: hidden;">
   <img src="/junzhin-homepage/assets/img/surgical_ophora.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
    <div class="text-justified">
      In ophthalmic surgery, developing an AI system capable of interpreting surgical videos and predicting subsequent operations requires numerous ophthalmic surgical videos with high-quality annotations, which are difficult to collect due to privacy concerns and labor consumption. Text-guided video generation (T2V) emerges as a promising solution to overcome this issue by generating ophthalmic surgical videos based on surgeon instructions. In this paper, we present Ophora, a pioneering model that can generate ophthalmic surgical videos following natural language instructions. To construct Ophora, we first propose a Comprehensive Data Curation pipeline to convert narrative ophthalmic surgical videos into a large-scale, high-quality dataset comprising over 160K video-instruction pairs, Ophora-160K. Then, we propose a Progressive Video-Instruction Tuning scheme to transfer rich spatial-temporal knowledge from a T2V model pre-trained on natural video-text datasets for privacy-preserved ophthalmic surgical video generation based on Ophora-160K. Experiments on video quality evaluation via quantitative analysis and ophthalmologist feedback demonstrate that Ophora can generate realistic and reliable ophthalmic surgical videos based on surgeon instructions.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **Advancing Medical Image Grounding via Spatial-Semantic Rewarded Group Relative Policy Optimization**  
  _MICCAI 2025, Co-author, Early Accepted_
  
  <div style="overflow: hidden;">
   <img src="/junzhin-homepage/assets/img/spatial_reward_r1_cxr.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
    <div class="text-justified">
      Medical Image Grounding (MIG), which involves localizing specific regions in medical images based on textual descriptions, requires models to not only perceive regions but also deduce spatial relationship of these regions. Existing Vision-Language Models (VLMs) for MIG often rely on Supervised Fine-Tuning (SFT) with large amounts of Chain-of-Thought (CoT) reasoning annotations, which are expensive and time-consuming to acquire. Recently, DeepSeek-R1 demonstrated that Large Language Models (LLMs) can acquire reasoning abilities through Group Relative Policy Optimization (GRPO) without requiring CoT annotations. In this paper, we adapt the GRPO reinforcement learning framework to VLMs for Medical Image Grounding. We propose the Spatial-Semantic Rewarded Group Relative Policy Optimization to train the model without CoT reasoning annotations. Specifically, we introduce Spatial-Semantic Rewards, which combine spatial accuracy reward and semantic consistency reward to provide nuanced feedback for both spatially positive and negative completions. Additionally, we propose to use the Chain-of-Box template, which integrates visual information of referring bounding boxes into the <think> reasoning process, enabling the model to explicitly reason about spatial regions during intermediate steps. Experiments on three datasets MS-CXR, ChestX-ray8, and M3D-RefSeg—demonstrate that our method achieves state-of-the-art performance in Medical Image Grounding.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **Multi-modal MRI Translation via Evidential Regression and Distribution Calibration**  
  _MICCAI 2025, Co-author, Early Accepted_
  
  <div style="overflow: hidden;">
   <img src="/junzhin-homepage/assets/img/multi-model_mri.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
    <div class="text-justified">
      Multi-modal Magnetic Resonance Imaging (MRI) translation leverages information from source MRI sequences to generate target modalities, enabling comprehensive diagnosis while overcoming the limitations of acquiring all sequences. While existing deep-learning-based multi-modal MRI translation methods have shown promising potential, they still face two key challenges: 1) lack of reliable uncertainty quantification for synthesized images, and 2) limited robustness when deployed across different medical centers. To address these challenges, we propose a novel framework that reformulates multi-modal MRI translation as a multi-modal evidential regression problem with distribution calibration. Our approach incorporates two key components: 1) an evidential regression module that estimates uncertainties from different source modalities and an explicit distribution mixture strategy for transparent multi-modal fusion, and 2) a distribution calibration mechanism that adapts to source-target mapping shifts to ensure consistent performance across different medical centers. Extensive experiments on three datasets from the BraTS2023 challenge demonstrate that our framework achieves superior performance and robustness across domains.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **Deep Generative Models Unveil Patterns in Medical Images Through Vision-Language Conditioning**  
  _AIM-FM Workshop at NeurIPS'24, First co-author, Oral Presentation, Accepted and In Press_ [Code](https://github.com/junzhin/DGM-VLC) [Paper](http://arxiv.org/abs/2410.13823)
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/mask2ct.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
    <div class="text-justified">
    This project explores how deep generative models can go beyond traditional data augmentation in medical imaging by uncovering and demonstrating clinical patterns within medical images. By integrating clinical data and segmentation masks, we guide the image synthesis process, transforming tabular clinical data into textual descriptions for easier interpretation. Our approach leverages large pre-trained vision-language models to capture relationships between clinical information and images, enabling more precise visualization of complex conditions. This method, applicable to both GAN-based and diffusion models, offers new potential for early detection and visualization of subtle clinical attributes in medical images.
    <div>
  <div> 

  <div style="clear: both;"></div>

- **DMRN: A Dynamical Multiorder Response Network for Robust Lung Airway Segmentation**  
  _WACV 2025, Co-author, Accepted and In Press_
    <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/DMRN.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
      <div class="text-justified">
       Automated airway segmentation in CT images is crucial for lung disease diagnosis. However, manual annotation scarcity hinders supervised learning efficacy, while unlimited intensities and sample imbalance lead to discontinuity and false-negative issues. To address these challenges, we propose a novel airway segmentation model named Dynamical Multi-order Response Network (DMRN), integrating unsupervised and supervised learning in parallel to alleviate the label scarcity of airway. In the unsupervised branch, we propose several novel strategies of Dynamic Mask-Ratio (DMR) to enable the model to perceive context information of varying sizes, mimicking the laws of human learning; we also present a novel target of Multi-Order Normalized Responses (MONR), exploiting the distinct order exponential operation of raw images and oriented gradients to enhance the textural representations of bronchioles. For the supervised branch, we directly predict the final full segmentation map by the large-ratio cube-masked input instead of full input. Ultimately, we verify the method's performance and robustness by training on normal lung disease datasets, while testing on lung cancer, COVID-19, and lung fibrosis datasets. All experimental results prove that our method exceeds state-of-the-art methods significantly. Code will be released in the future.
      <div>
    <div> 
  
- **Decoding Report Generators: Cyclic Vision-Language Adapter for Counterfactual Explanations**  
  _IJCAI 2025, Co-author, Accepted_ 
   <div> 
    <img src="/junzhin-homepage/assets/img/Cyclic_Vision_LanguageManipulator.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
      <div class="text-justified">
        Despite significant advancements in report generation methods, a critical limitation remains: the lack of interpretability in the generated text. This paper introduces an innovative approach to enhance the explainability of text generated by report generation models. Our method employs cyclic text manipulation and visual comparison to identify and elucidate the features in the original content that influence the generated text. By manipulating the generated reports and producing corresponding images, we create a comparative framework that highlights key attributes and their impact on the text generation process. This approach not only identifies the image features aligned to the generated text but also improves transparency but also provides deeper insights into the decision-making mechanisms of the report generation models. Our findings demonstrate the potential of this method to significantly enhance the interpretability and transparency of AI-generated reports.
      <div>
    <div> 
  
- **UNVEILING THE CAPABILITIES OF LATENT DIFFUSION MODELS FOR CLASSIFICATION OF LUNG DISEASES IN CHEST X-RAYS**  
  _ISBI 2025,** First author**, Accepted_ 
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/latent_diffusion_cxr.png" alt="Latent Diffusion Models for CXR Classification" style="float: left; margin-right: 20px; width: 150px;">
    
    <div class="text-justified">
      Diffusion models have demonstrated a remarkable ability to synthesize Chest X-Ray (CXR) images, particularly by generating high-quality samples to address the scarcity and imbalance of annotated CXRs. However, while they excel in sample generation, leveraging their discriminative capabilities for disease classification remains a challenge. This project investigates an approach that utilizes latent conditional diffusion models, conditioned on corresponding radiology reports, to classify lung disease patterns observed in CXRs. Specifically, a pre-trained latent conditional diffusion model is employed to predict noise estimates for a noisy input lung image under different disease conditions. By comparing the noise estimation errors associated with various class prompts, the most probable classification is determined based on minimal error. This method not only facilitates effective classification but also enhances the interpretability of the generative model during training. Experimental results demonstrate that the diffusion-based classifier achieves zero-shot classification performance comparable to existing models, with the regions identified by the model corresponding to actual lesion areas in CXRs.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **ARMUT-LOR: Adaptive Region-aware Masked Unpaired Translation for Lung Opacity Removal in Chest X-rays**  
  _Master Research Project, Imperial College London, First author_ [Thesis](https://drive.google.com/file/d/1Ue34uFvl31JfXWE4U1YgNUjGLe6b0SDd/view?usp=sharing)
  
  <img src="/junzhin-homepage/assets/img/imperial.png" alt="ARMUT-LOR" style="float: left; margin-right: 20px; width: 150px;">
  
  <div style="clear: both;"></div>

- **Unpaired Translation of Chest X-ray Images for Lung Opacity Diagnosis via Adaptive Activation Masks**  
  _Pattern Recognition Letters, First author, Accepted_ [Graphical Abstract](https://drive.google.com/file/d/1337JIJPOp26r9chK91QP5FJkll67aT7-/view?usp=drive_link)  
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/cxr_lung_opacity.png" alt="Unpaired Translation" style="float: left; margin-right: 20px; width: 150px;">

    <div class="text-justified">
    This project addresses the challenge of diagnosing cardiopulmonary diseases in chest X-rays (CXRs) when lung opacities obscure critical anatomical details. To improve the clarity of these images, we propose a novel unpaired CXR translation framework that converts CXRs with opacities into versions without them, while preserving essential diagnostic features. Our method uses adaptive activation masks to selectively modify opacity-affected regions and aligns the translated images with pre-trained models to ensure clinical relevance. This approach aims to enhance the accuracy of lung border segmentation and lesion detection, improving CXR interpretation in medical practice.
    <div>
  <div>

  <div style="clear: both;"></div>

- **Anatomy-Guided Radiology Report Generation with Pathology-Aware Regional Prompts**  
  _JBHI 2024, Co-author, Under Review_

- **Enhancing Night-to-Day Image Translation with Semantic Prior and Reference Image Guidance**  
  _Australasian Database Conference (ADC 2023), First author_ [Link](https://link.springer.com/chapter/10.1007/978-3-031-47843-7_12), [Slides](https://drive.google.com/file/d/1gfHnrjvkWIF_IKdZ749TUwpAdwDYwy05/view?usp=drive_link)  
  
  <div style="overflow: hidden;">
  
    <img src="/junzhin-homepage/assets/img/semantic.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 150px;">

    <div class="text-justified">
      This project introduces "RefN2D-Guide GAN," a novel method for improving night-to-day image translation by addressing the challenge of mapping images between domains with varying information richness. By incorporating reference images and feature matching loss, our approach enhances the adaptability of the generator's encoder. Additionally, a segmentation module helps preserve semantic details in the translated images without requiring ground truth annotations. This technique ensures better quality in night-to-day translations while maintaining semantic consistency, offering improvements in tasks like autonomous driving and scene understanding.
    </div>

  </div>
  
  <div style="clear: both;"></div>

 
## Selected Projects and Course Work


- **Multi-class Classification using Text-Vision Fusion Model**  
[Report](https://drive.google.com/file/d/18U67_H5nPL6BjajxfbX34E5AqhIBEl3S/view?usp=sharing), [Poster](https://drive.google.com/file/d/1oYXzy0yNNpW3BrtD8QGXR1eBkmAVe_2B/view?usp=drive_link) 

    <div> 
    <img src="/junzhin-homepage/assets/img/text_lan_fusion.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 200px;">
    <div class="text-justified"> 
     Designed a multimodal model integrating EfficientNet for image features and BERT for text features. Conducted experiments on hyperparameters and modules, performing an ablation study of vision and text models, and optimizing learning rate, dropout, and other key factors.  

    <div>
  <div>  

  <div style="clear: both;"></div>


- **Multi-class Classification using self-implemented Multilayer Perceptron**  
[Code](https://github.com/junzhin/DL_assign1/tree/main),[Report](https://github.com/junzhin/DL_assign1/blob/main/reports/490059823_520576076_ver6_0.pdf) 
  <div> 
    <img src="/junzhin-homepage/assets/img/self_implemented_mlp.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 200px;">
    <div class="text-justified"> 
     Developed an MLP from scratch using NumPy for a 10-class classification task. Experimented with activation functions, optimizers, and regularization techniques, conducting ablation studies to refine model performance.  

    <div>
  <div> 

  <div style="clear: both;"></div>

 
- **Robustness of Methods for Class-Dependent Label Noise** 
 [Report](https://junzhin.github.io/junzhin-homepage/)
  <div> 
    <img src="/junzhin-homepage/assets/img/noise4.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 200px;">
    <img src="/junzhin-homepage/assets/img/noise1.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 200px;">
    <div class="text-justified">  
    Reproduced paradigms of class-dependent label noise using PyTorch, and conducted experiments with forward learning and weighted loss techniques to estimate the flip rate matrix.  

    <div>
  <div> 

  <div style="clear: both;"></div>

  
- **Visialisation How did the mobile device market change?** 
[Link](https://junzhin.github.io/junzhin-homepage/) 
  <div> 
    <img src="/junzhin-homepage/assets/img/va_1.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 200px;">
    <div class="text-justified">  
    <img src="/junzhin-homepage/assets/img/va_2.png" alt="Night-to-Day Image Translation" style="float: left; margin-right: 20px; width: 200px;">
    <div class="text-justified">  
  This project visualizes the evolution of the mobile device market from 1989 to 2012 using Tableau. Key steps include data preprocessing, extracting company names, and categorizing device types over time. The visualization highlights market trends, dominant companies, and changes in device technology.
    <div>
  <div>  

  <div style="clear: both;"></div>

- **Rethinking ImageNet Pretraining in Domain Adaptation**  
[Link](https://junzhin.github.io/junzhin-homepage/)
  <div> 
    <div class="text-justified">
      Analyzed the impact of ImageNet pretraining on semi-supervised and unsupervised domain adaptation tasks using PyTorch models. Conducted large-scale computations in the cloud, proposing how pre-trained neural networks influence migration learning problems.  

    <div>
  <div> 

  <div style="clear: both;"></div>

- **Other sample of work**
[To download](https://drive.google.com/file/d/18U67_H5nPL6BjajxfbX34E5AqhIBEl3S/view?usp=sharing)


## Extra

A passionate fan of documentaries.
I enjoy strategy games, especially grand strategy games from Paradox Interactive.
I like playing table tennis and practicing calligraphy.
Always curious and love experimenting with new ideas. 


## Contact
- [**Email**](mailto:ningjunzhi85@gmail.com)
- [**LinkedIn**](https://www.linkedin.com/in/junzhin)
- [**Scholar**](https://scholar.google.com/citations?hl=en&user=-yPon1YAAAAJ)

