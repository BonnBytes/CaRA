Canonical Rank Approximation (CaRA): An Efficient Fine-Tuning Strategy for Vision Transformers
**********************************************************************************************

`Lokesh Veeramacheneni <https://lokiv.dev>`__\ :sup:`1`, `Moritz
Wolter <https://www.wolter.tech/>`__\ :sup:`1`, `Hilde
Kuehne <https://hildekuehne.github.io/>`__\ :sup:`2`, and `Juergen
Gall <https://pages.iai.uni-bonn.de/gall_juergen/>`__\ :sup:`1,3`

| 1. *University of Bonn* 
| 2. *University of Tübingen, MIT-IBM Watson AI Lab*
| 3. *Lamarr Institute for Machine Learning and Artificial Intelligence*
|


|License| |Arxiv|  |Project|

**Keywords:** CaRA, Canonical Polyadic Decomposition, CPD, Tensor methods, ViT, LoRA 

**Abstract:** Modern methods for fine-tuning a Vision Transformer (ViT) like Low-Rank Adaptation (LoRA) and its variants demonstrate impressive performance. However, these methods ignore the high-dimensional nature of Multi-Head Attention (MHA) weight tensors. To address this limitation, we propose Canonical Rank Adaptation (CaRA). CaRA leverages tensor mathematics, first by tensorising the transformer into two different tensors; one for projection layers in MHA and the other for feed-forward layers. Second, the tensorised formulation is fine-tuned using the low-rank adaptation in Canonical-Polyadic Decomposition (CPD) form. Employing CaRA efficiently minimizes the number of trainable parameters. Experimentally, CaRA outperforms existing Parameter-Efficient Fine-Tuning (PEFT) methods in visual classification benchmarks such as Visual Task Adaptation Benchmark (VTAB)-1k and Fine-Grained Visual Categorization (FGVC).


.. image:: https://raw.githubusercontent.com/BonnBytes/CaRA/refs/heads/main/images/tensorisation.jpg
   :width: 100%
   :alt: Alternative text


Installation
============

Use `UV <https://docs.astral.sh/uv/>`_ to install the requirements

For CPU based pytorch

.. code:: bash

   uv sync --extra cpu 

For CUDA based pytorch

.. code:: bash

   uv sync --extra cu118


Datasets
=======

In the case of VTAB-1k benchmark, refer to the dataset download instructions from `NOAH <https://github.com/ZhangYuanhan-AI/NOAH>`_. We download the datasets for FGVC benchmark from their respective sources.

Note: Create a ``data`` folder in the root and place the datasets inside this folder.


Pretrained models
=================
Please refer to the download links provided in the paper.


Training
========
For fine-tuning ViT use the following command.

.. code:: bash
   
   export PYTHONPATH=.
   python image_classification/vit_cp.py --dataset=<choice_of_dataset> --dim=<rank>


Evaluation
==========
We provide the link for fine-tuned models for each dataset in VTAB-1k benchmark `here <https://uni-bonn.sciebo.de/s/YAtcRDHxdwnBGq7>`_. To reproduce results from the paper, download the model and execute the following command

.. code:: bash

   export PYTHONPATH=.
   python image_classification/vit_cp.py --dataset=<choice_of_dataset> --dim=<rank> --evaluate=<path_to_model>


Acknowledgments
===============
The code is built on the implementation of `FacT <https://github.com/JieShibo/PETL-ViT/tree/main/FacT>`__. Thanks to `Zahra Ganji <https://github.com/ZahraGanji>`__ for reimplementing VeRA baseline.



.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |Project| image:: https://img.shields.io/badge/Project-Website-blue
   :target: https://lokiv.dev/cara/
   :alt: Project Page
.. |Arxiv| image:: https://img.shields.io/badge/OpenReview-Paper-blue
   :target: https://openreview.net/pdf?id=vexHifrbJg
   :alt: Paper
