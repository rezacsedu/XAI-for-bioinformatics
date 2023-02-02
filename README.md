# Explainale AI (XAI) for bioinformatics
Codes and supplementary materials for our paper "Explainable AI for Bioinformatics: Importance, Methods, Tools, and Applications", submitted to [Briefings in Bioinformatics](https://academic.oup.com/bib) journal. This repo will be updated periodically.

## Notebooks
We provided several interactive jupyter notebooks showing how interpretable ML techniques can be used to improve the interpretability for bioinformatics research use cases. Please note that the notebooks don't accompany the dataset, which is mainly NDA agreements. 

## Paers and books on interpretable ML methods
We categorize the papers and books based on interpretable ML methods

### Books
* A Guide for Making Black Box Models Explainable. _Molnar 2019_ [pdf](https://christophm.github.io/interpretable-ml-book/)

### Surveys (papers)
* A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI. _Tjao et al. 2020_ [pdf](https://arxiv.org/pdf/1907.07374.pdf)
* Opportunities and Challenges in Explainable Artificial Intelligence (XAI): A Survey. _Das et al. 2020_ [pdf](https://arxiv.org/pdf/2006.11371.pdf)
* Interpretable machine learning: definitions, methods, and applications. _Murdoch et al. 2019_ [pdf](https://arxiv.org/pdf/1901.04592v1.pdf)
* A brief survey of visualization methods for deep learning models from the perspective of Explainable AI. _Chalkiadakis 2018_ [pdf](https://www.macs.hw.ac.uk/~ic14/IoannisChalkiadakis_RRR.pdf)
* A Survey Of Methods For Explaining Black Box Models. _Guidotti et al. 2018_ [pdf](https://arxiv.org/pdf/1802.01933.pdf)
* Explaining Explanations: An Overview of Interpretability of Machine Learning. _Gilpin et al. 2019_ [pdf](https://arxiv.org/pdf/1806.00069.pdf)
* Explainable Artificial Intelligence: a Systematic Review. _Vilone at al. 2020_ [pdf](https://arxiv.org/pdf/2006.00093.pdf)

### Attribution maps and gradient-based (papers)
* `DTCAV`: Automating Interpretability: Discovering and Testing Visual Concepts Learned by Neural Networks. _Ghorbani et al. 2019_ [pdf](https://arxiv.org/abs/1902.03129)
* `AM`: Visualizing higher-layer features of a deep network. _Erhan et al. 2009_ [pdf](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network)
* Deep inside convolutional networks: Visualising image classification models and saliency maps. _Simonyan et al. 2013_ [pdf](https://arxiv.org/pdf/1312.6034.pdf)
* `DeepVis`: Understanding Neural Networks through Deep Visualization. _Yosinski et al. ICML workshop 2015_ [pdf](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf) 
* Visualizing and Understanding Recurrent Networks. _Kaparthey et al. ICLR 2015_ [pdf](https://arxiv.org/abs/1506.02078)
* Feature Removal Is A Unifying Principle For Model Explanation Methods. _Covert et al. 2020_ [pdf](https://arxiv.org/pdf/2011.03623.pdf) 
* `Gradient`: Deep inside convolutional networks: Visualising image classification models and saliency maps. _Simonyan et al. 2013_ [pdf](https://arxiv.org/pdf/1312.6034.pdf)
* `Guided-backprop`: Striving for simplicity: The all convolutional net. _Springenberg et al. 2015_ [pdf](http://arxiv.org/pdf/1412.6806.pdf)
* `SmoothGrad`: removing noise by adding noise. _Smilkov et al. 2017_ [pdf](https://arxiv.org/abs/1706.03825)
* `DeepLIFT`: Learning important features through propagating activation differences. _Shrikumar et al. 2017_ [pdf](https://arxiv.org/pdf/1605.01713.pdf)
* `IG`: Axiomatic Attribution for Deep Networks. _Sundararajan et al. 2018_ [pdf](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) 
* `EG`: Learning Explainable Models Using Attribution Priors. _Erion et al. 2019_ [pdf](https://arxiv.org/abs/1906.10670) 
* `LRP`: Beyond saliency: understanding convolutional neural networks from saliency prediction on layer-wise relevance propagation [pdf](https://arxiv.org/abs/1712.08268)
* `DTD`: Explaining NonLinear Classification Decisions With Deep Tayor Decomposition [pdf](https://arxiv.org/abs/1512.02479)
* `CAM`: Learning Deep Features for Discriminative Localization. _Zhou et al. 2016_ [link](http://cnnlocalization.csail.mit.edu/)
* `Grad-CAM`: Visual Explanations from Deep Networks via Gradient-based Localization. _Selvaraju et al. 2017_ [pdf](https://arxiv.org/abs/1610.02391)
* `Grad-CAM++`: Improved Visual Explanations for Deep Convolutional Networks. _Chattopadhyay et al. 2017_ [pdf](https://arxiv.org/abs/1710.11063) 
* `Smooth Grad-CAM++`: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models. _Omeiza et al. 2019_ [pdf](https://arxiv.org/pdf/1908.01224.pdf)
* `NormGrad`: There and Back Again: Revisiting Backpropagation Saliency Methods. _Rebuffi et al. CVPR 2020_ [pdf](https://arxiv.org/abs/2004.02866) 
* `Score-CAM`: Score-Weighted Visual Explanations for Convolutional Neural Networks. _Wang et al. CVPR 2020 workshop_ [pdf](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf)
* `Relevance-CAM`: Your Model Already Knows Where to Look. _Lee et al. CVPR 2021_ [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Relevance-CAM_Your_Model_Already_Knows_Where_To_Look_CVPR_2021_paper.pdf) 
* `LIFT-CAM`: Towards Better Explanations of Class Activation Mapping. _Jung & Oh ICCV 2021_ [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Jung_Towards_Better_Explanations_of_Class_Activation_Mapping_ICCV_2021_paper.pdf).

### Sensitivity and perturbation-based (papers)
* Generative causal explanations of black-box classifiers. _O’Shaughnessy et al. 2020_ [pdf](https://arxiv.org/abs/2006.13913) 
* Removing input features via a generative model to explain their attributions to classifier's decisions. _Agarwal et al. 2019_ [pdf](https://arxiv.org/abs/1910.04256) 
* Challenging common interpretability assumptions in feature attribution explanations? _Dinu et al. NeurIPS workshop 2020_ [pdf](https://arxiv.org/abs/2012.02748)
* The effectiveness of feature attribution methods and its correlation with automatic evaluation scores. _Nguyen, Kim, Nguyen 2021_ [pdf](http://anhnguyen.me/project/feature-attribution-effectiveness/)
* `Deletion` & `Insertion`: Randomized Input Sampling for Explanation of Black-box Models. _Petsiuk et al. BMVC 2018_ [pdf](https://arxiv.org/pdf/1806.07421.pdf)
* DiffROAR: Do Input Gradients Highlight Discriminative Features? _Shah et al. NeurIPS 2021_ [pdf](https://arxiv.org/pdf/2102.12781.pdf)
* `RISE`: Randomized Input Sampling for Explanation of Black-box Models. _Petsiuk et al. BMVC 2018_ [pdf](https://arxiv.org/pdf/1806.07421.pdf)
* `LIME`: Why should i trust you?: Explaining the predictions of any classifier. _Ribeiro et al. 2016_ [pdf](https://arxiv.org/pdf/1602.04938.pdf) 
* `LIME-G`: Removing input features via a generative model to explain their attributions to classifier's decisions. _Agarwal & Nguyen. ACCV 2020_ [pdf](https://arxiv.org/abs/1910.04256)
* `SHAP`: A Unified Approach to Interpreting Model Predictions. _Lundberg et al. 2017_ [pdf](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
* `IM`: Interpretation of NLP models through input marginalization. _Kim et al. EMNLP 2020_ [pdf](https://arxiv.org/abs/2010.13984).

### Rule- and counterfactual explanations (papers)
* Local Rule-based Explanations of Black Box Decision Systems. _Guidotti et al. 2021_ [pdf](https://arxiv.org/pdf/1805.10820.pdf)
* `FIDO`: Explaining image classifiers by counterfactual generation. _Chang et al. ICLR 2019_ [pdf](https://arxiv.org/pdf/1807.08024.pdf)
* `CEM`: Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives. _Dhurandhar & Chen et al. NeurIPS 2018_ [pdf](https://proceedings.neurips.cc/paper/2018/file/c5ff2543b53f4cc0ad3819a36752467b-Paper.pdf)
* Counterfactual Explanations for Machine Learning: A Review. _Verma et al. 2020_ [pdf](https://arxiv.org/pdf/2010.10596.pdf)
* Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections. _Zhang et al. 2018_ [pdf](http://papers.nips.cc/paper/7736-interpreting-neural-network-judgments-via-minimal-stable-and-symbolic-corrections.pdf)
* Counterfactual Visual Explanations. _Goyal et al. 2019_ [pdf](https://arxiv.org/pdf/1904.07451.pdf)
* Generative Counterfactual Introspection for Explainable Deep Learning. _Liu et al. 2019_ [pdf](https://arxiv.org/abs/1907.03077).

### Knowledge-based (papers)
* ReasonChainQA: Text-based Complex Question Answering with Explainable Evidence Chains. _Zhu et al. 2022_ [pdf](https://arxiv.org/pdf/2210.08763.pdf)
* Knowledge-graph-based explainable AI: A systematic review. _Rajabi et al. 2022_ [link](https://journals.sagepub.com/doi/full/10.1177/01655515221112844)
* Knowledge-based XAI through CBR: There is more to explanations than models can tell. _Weber et al. 2021_ [pdf](https://arxiv.org/pdf/2108.10363.pdf)
* The Role of Human Knowledge in Explainable AI. _Tocchetti  et al. 2022_ [link](https://www.mdpi.com/2306-5729/7/7/93).

### XAI with focus on HCI (papers)
* Question-Driven Design Process for Explainable AI User Experiences _Liao 2021_ [pdf](https://arxiv.org/pdf/2104.03483.pdf)
* Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior? _Hase & Bansal ACL 2020_ [pdf](https://arxiv.org/pdf/2005.01831.pdf) 
* Teach Me to Explain: A Review of Datasets for Explainable NLP. _Wiegreffe & Marasović 2021_ [pdf](https://arxiv.org/abs/2102.12060)
* Yang, S. C. H., & Shafto, P. Explainable Artificial Intelligence via Bayesian Teaching. NIPS 2017 [pdf](http://shaftolab.com/assets/papers/yangShafto_NIPS_2017_machine_teaching.pdf)
* Explainable AI for Designers: A Human-Centered Perspective on Mixed-Initiative Co-Creation [pdf](http://www.antoniosliapis.com/papers/explainable_ai_for_designers.pdf)
* ICADx: Interpretable computer aided diagnosis of breast masses. _Kim et al. 2018_ [pdf](https://arxiv.org/abs/1805.08960)
* Neural Network Interpretation via Fine Grained Textual Summarization. _Guo et al. 2018_ [pdf](https://arxiv.org/pdf/1805.08969.pdf)
* LS-Tree: Model Interpretation When the Data Are Linguistic. _Chen et al. 2019_ [pdf](https://arxiv.org/abs/1902.04187).

### Distilling DNNs into more interpretable models (papers)
* Interpreting CNNs via Decision Trees [pdf](https://arxiv.org/abs/1802.00121)
* Distilling a Neural Network Into a Soft Decision Tree [pdf](https://arxiv.org/abs/1711.09784)
* Improving the Interpretability of Deep Neural Networks with Knowledge Distillation. _Liu et al. 2018_ [pdf](https://arxiv.org/pdf/1812.10924.pdf).

## Application areas
### Computer Vision
* Multimodal explanations: Justifying decisions and pointing to the evidence. _Park et al. CVPR 2018_ [pdf](https://arxiv.org/abs/1802.08129)
* `IA-RED2`: Interpretability-Aware Redundancy Reduction for Vision Transformers. _Pan et al. NeurIPS 2021_ [pdf](https://arxiv.org/abs/2106.12620)
* Transformer Interpretability Beyond Attention Visualization. _Hila et al. CVPR 2021_ [pdf](https://arxiv.org/abs/2012.09838)

### NLP
* `Deletion_BERT`: Double Trouble: How to not explain a text classifier’s decisions using counterfactuals synthesized by masked language models. _Pham et al. 2022_ [pdf](https://arxiv.org/abs/2110.11929) 
* Considering Likelihood in NLP Classification Explanations with Occlusion and Language Modeling. _Harbecke et al. 2020_ [pdf](https://arxiv.org/abs/2004.09890).

## Interpretable ML tools and libraries
### GUI tools
* `DeepVis`: Deep Visualization Toolbox. _Yosinski et al. ICML 2015_ [code](https://github.com/yosinski/deep-visualization-toolbox) 
* `SWAP`: Generate adversarial poses of objects in a 3D space. _Alcorn et al. CVPR 2019_ [code](https://github.com/airalcorn2/strike-with-a-pose) 
* `AllenNLP`: Query online NLP models with user-provided inputs and observe explanations (Gradient, Integrated Gradient, SmoothGrad). _Last accessed 03/2020_ [demo](https://demo.allennlp.org/sentiment-analysis)
* `3DB`: A framework for analyzing computer vision models with simulated data [code](https://github.com/3db/3db/).

### Libraries
* [CNN visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (feature visualization, PyTorch)
* [iNNvestigate](https://github.com/albermax/innvestigate) (attribution, Keras)
* [DeepExplain](https://github.com/marcoancona/DeepExplain) (attribution, Keras)
* [Lucid](https://github.com/tensorflow/lucid) (feature visualization, attribution, Tensorflow)
* [TorchRay](https://facebookresearch.github.io/TorchRay/) (attribution, PyTorch)
* [Captum](https://captum.ai/) (attribution, PyTorch)
* [InterpretML](https://github.com/interpretml/interpret) (attribution, Python).

## Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @article{karim_xai_bio_2022,
          title={Explainable AI for Bioinformatics: Methods, Tools, and Applications},
          author={Karim, Md Rezaul and Beyan, Oya and Zappa, Achille and Costa, Ivan G and Rebholz-Schuhmann, Dietrich and Cochez, Michael and Decker, Stefan},
          journal={Briefings in bioinformatics},
          volume={XXXX},
          number={XXXX},
          pages={XXXX},
          year={2022},
          publisher={Oxford University Press}
          }

## Contributing
If you find more related work, which are not listed here, please create a PR or sugest by filing issues. Your contribution will be highly appreciated. For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de. 
