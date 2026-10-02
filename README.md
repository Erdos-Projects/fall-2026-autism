# autismCNN
Model identifying autism individuals by MRI and/or fMRI.

## Problem Definition
> What decision or action will this analysis inform?

This analysis will inform the viability of training a CNN model to support medical professionals in diagnosing Autism. 

> Who are the stakeholders and what do they care about?

This will help radiologists, doctors, therapists, and patients by adding an unbiased source of diagnostic criteria as well as speeding up the time to diagnose autism.

> Specify the unit of analysis

We will be using the data from the Autism Brain Imaging Data Exchange (ABIDE) initiative. 

> Define the scope and boundaries

**Time horizon:** 2012

**Geographic region:** USA, Ireland, Belgium, France, Switzerland, Netherlands

**Population:** 1112 total, 573 control, 539 autism

**Features included:** MRI scans, fMRI scans, age group, sex, slice thickness, weighting, pixel size of images, field strength, acquisition type, acquisition plane, flip angle, manufacturer, image size (pixels), pixel spacing, autism type. 

>Identify anti-goals

We will not be finding the cause of autism.

## Data Gathering

**Source:** http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html; https://ida.loni.usc.edu/home/projectPage.jsp?project=ABIDE (login required)

**Acquisition strategy:** One-time download from website.

**Database queries:** Research group: "all"; Sex: "both", Modality: "fMRI" & "MRI", Image count: "ALL"

**Privacy:** All data is anonymous and HIPPA compliant. From the website:

> Consistent with the policies of the 1000 Functional Connectomes Project, data usage is unrestricted for non-commercial research purposes. We kindly request that the specific datasets included in analyses be specified appropriately, and that their funding sources be acknowledged. As per INDI protocol, we simply require that users register with the NITRC and 1000 Functional Connectomes Project to gain access. (Creative Commons, Attribution-NonCommercial-Share Alike License)



## Data Assessment

**Volume:** The sample size in on par with other studies.
  
**Granularity:** We have the full demographic information and medical scans.

**Bias and representativeness:** The dataset only includes Western Europe and the United States. The demographic information also does not include ethnicity. 

## Assessing Learnability

> Do the features plausibly contain information about the target?

The autism and control scans are labeled. 

> Are there enough examples overall and per class?

The number of samples for autism and control are roughly half and half. 

## KPI Definition 

- **Primary KPI:** accuracy
- **Secondary KPI:** fairness metrics -- accuracy based on age and/or gender
