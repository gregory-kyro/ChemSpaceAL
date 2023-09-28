![logo](https://github.com/gregory-kyro/ChemSpaceAL/assets/98780179/79f0f8cf-6f0a-45cf-85b2-854f6c8ff7d5)

ChemSpaceAL: An Efficient Active Learning Methodology Applied to Protein-Specific Molecular Generation

![toc_figure](https://github.com/gregory-kyro/ChemSpaceAL/assets/98780179/ebdcdec1-67d0-48e5-92f1-330ab921b42b)

## Summary
The incredible capabilities of generative artificial intelligence models have inevitably led to their application in the domain of drug discovery. It is therefore of tremendous interest to develop methodologies that enhance the abilities and applicability of these powerful tools. In this work, we present a novel and efficient semi-supervised active learning methodology that allows for the fine-tuning of a generative model with respect to an objective function by strategically operating within a constructed representation of the sample space. In the context of targeted molecular generation, we demonstrate the ability to fine-tune a GPT-based molecular generator with respect to an attractive interaction-based scoring function by strategically operating within a chemical space proxy, thereby maximizing attractive interactions between the generated molecules and a protein target. Importantly, our approach does not require the individual evaluation of all data points that are used for fine-tuning, enabling the incorporation of computationally expensive metrics. We are hopeful that the inherent generality of this methodology ensures that it will remain applicable as this exciting field evolves. To facilitate implementation and reproducibility, we have made all of our software available through the open-source ChemSpaceAL Python package.

## Tutorial Notebook Using HACNet Python Package
https://colab.research.google.com/github/gregory-kyro/ChemSpaceAL/blob/main/ChemSpaceAL.ipynb

## Associated Preprint
https://pubmed.ncbi.nlm.nih.gov/37744464/

## Python Package
https://pypi.org/project/ChemSpaceAL/

in order to install the ChemSpaceAL package, simply run:

```pip install ChemSpaceAL```

## Contact
Please feel free to reach out to us through either of the following emails if you have any questions or need any additional files:

gregory.kyro@yale.edu

anton.morgunov@yale.edu

rafi.brent@yale.edu
