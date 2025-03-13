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

- (Oct 2023 - Oct 2024) **Research Graduate Assistant** – Imperial College London
- (Aug 2022 - Jun 2023) **Research Undergraduate Assistant** – Trustworthy Machine Learning Lab, University of Sydney
- (Nov 2022 - Jan 2023) **Software Engineering Intern** – Guangzhou Junrui Kang Biotechnology
- (Dec 2021 - Jun 2022) **Research Undergraduate Assistant** – University of Melbourne




## Selected Publication and Research Work

- **RetinaLogos: Fine-Grained Synthesis of High-Resolution Retinal Images Through Captions**  
  _(Under Review)_ [Code](https://anonymous.4open.science/r/Text-Driven-CFP-Generator)
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/retina_logos.png" alt="RetinaLogos" style="float: left; margin-right: 20px; width: 150px;">
    
    <div class="text-justified">
      The scarcity of high-quality, labelled retinal imaging data presents a significant challenge in the development of machine learning models for ophthalmology. To address this, we introduce RetinaLogos-1400k, a large-scale synthetic Caption-CFP dataset with 1.4 million entries that uses large language models to describe retinal conditions and key structures. Based on this dataset, we employ a novel three-step training framework that enables fine-grained semantic control over retinal images and accurately captures different stages of disease progression and anatomical variations. Extensive experiments show that 62.07% of our text-driven synthetic images are indistinguishable from real ones by ophthalmologists, and the synthetic data improves accuracy by 10%-25% in diabetic retinopathy grading and glaucoma detection.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **UNVEILING THE CAPABILITIES OF LATENT DIFFUSION MODELS FOR CLASSIFICATION OF LUNG DISEASES IN CHEST X-RAYS**  
  _(Submission Under Review)_ ISBI 2025 [Paper](https://junzhin.github.io/junzhin-homepage/)
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/latent_diffusion_cxr.png" alt="Latent Diffusion Models for CXR Classification" style="float: left; margin-right: 20px; width: 150px;">
    
    <div class="text-justified">
      Diffusion models have demonstrated a remarkable ability to synthesize Chest X-Ray (CXR) images, particularly by generating high-quality samples to address the scarcity and imbalance of annotated CXRs. However, while they excel in sample generation, leveraging their discriminative capabilities for disease classification remains a challenge. This project investigates an approach that utilizes latent conditional diffusion models, conditioned on corresponding radiology reports, to classify lung disease patterns observed in CXRs. Specifically, a pre-trained latent conditional diffusion model is employed to predict noise estimates for a noisy input lung image under different disease conditions. By comparing the noise estimation errors associated with various class prompts, the most probable classification is determined based on minimal error. This method not only facilitates effective classification but also enhances the interpretability of the generative model during training. Experimental results demonstrate that the diffusion-based classifier achieves zero-shot classification performance comparable to existing models, with the regions identified by the model corresponding to actual lesion areas in CXRs.
    </div>
  </div>
  
  <div style="clear: both;"></div>

- **ARMUT-LOR: Adaptive Region-aware Masked Unpaired Translation for Lung Opacity Removal in Chest X-rays**  
  _(Master's Thesis)_ [Thesis](https://drive.google.com/file/d/1Ue34uFvl31JfXWE4U1YgNUjGLe6b0SDd/view?usp=sharing)
  
  <img src="/junzhin-homepage/assets/img/imperial.png" alt="ARMUT-LOR" style="float: left; margin-right: 20px; width: 150px;">
  
  <div style="clear: both;"></div>

- **Unpaired Translation of Chest X-ray Images for Lung Opacity Diagnosis via Adaptive Activation Mask**  
  _(Under Review)_  [Graphical Abstract](https://drive.google.com/file/d/1337JIJPOp26r9chK91QP5FJkll67aT7-/view?usp=drive_link)  
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/cxr_lung_opacity.png" alt="Unpaired Translation" style="float: left; margin-right: 20px; width: 150px;">

    <div class="text-justified">
    This project addresses the challenge of diagnosing cardiopulmonary diseases in chest X-rays (CXRs) when lung opacities obscure critical anatomical details. To improve the clarity of these images, we propose a novel unpaired CXR translation framework that converts CXRs with opacities into versions without them, while preserving essential diagnostic features. Our method uses adaptive activation masks to selectively modify opacity-affected regions and aligns the translated images with pre-trained models to ensure clinical relevance. This approach aims to enhance the accuracy of lung border segmentation and lesion detection, improving CXR interpretation in medical practice.
    <div>
  <div>

  <div style="clear: both;"></div>
  


- **Deep Generative Models Unveil Patterns in Medical Images Through Vision- “Language” Conditioning**  
  _AIM-FM Workshop (NeurIPS'24)_ [Code](https://github.com/junzhin/DGM-VLC) [Paper](http://arxiv.org/abs/2410.13823)
  
  <div style="overflow: hidden;">
    <img src="/junzhin-homepage/assets/img/mask2ct.png" alt="Deep Generative Models" style="float: left; margin-right: 20px; width: 150px;">
    <div class="text-justified">
    This project explores how deep generative models can go beyond traditional data augmentation in medical imaging by uncovering and demonstrating clinical patterns within medical images. By integrating clinical data and segmentation masks, we guide the image synthesis process, transforming tabular clinical data into textual descriptions for easier interpretation. Our approach leverages large pre-trained vision-language models to capture relationships between clinical information and images, enabling more precise visualization of complex conditions. This method, applicable to both GAN-based and diffusion models, offers new potential for early detection and visualization of subtle clinical attributes in medical images.
    <div>
  <div> 

  <div style="clear: both;"></div>

- **Enhancing Night-to-Day Image Translation with Semantic Prior and Reference Image Guidance**  
  _Australasian Database Conference (ADC 2023)_ [Link](https://link.springer.com/chapter/10.1007/978-3-031-47843-7_12), [Slides](https://drive.google.com/file/d/1gfHnrjvkWIF_IKdZ749TUwpAdwDYwy05/view?usp=drive_link)  
  
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

