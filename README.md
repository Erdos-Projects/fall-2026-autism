# autismCNN
Model identifying autism individuals by MRI and/or fMRI.

## Problem Definition
Clearly articulate the question you’re asking
- What decision or action will this analysis inform?
- Who are the stakeholders and what do they care about?

This analysis will inform the viability of training a CNN model to inform Autism diagnoses. This will help radiologists, doctors, therapists, and patients by adding an unbiased source of diagnostic criteria as well as speeding up the time to diagnose autism.

Specify the unit of analysis
- Individual, transaction, session, experiment, etc.

We will be using the data from the Autism Brain Imaging Data Exchange (ABIDE) initiative. 

Define the scope and boundaries
- Time horizon, geographic region, population, features included/excluded.

Time horizon: 2012, 2016
Geographic region: USA, Ireland, Belgium, France, Switzerland, Netherlands
Population: 1112 total, 573 control, 539 autism
Features included: MRI scans, fMRI scans, age group, sex, slice thickness, weighting, pixel size of images, field strength, acquisition type, acquisition plane, flip angle, manufacturer, image size (pixels), pixel spacing, autism type. 

Identify anti-goals
- Explicitly state what your project will not address.

We will not be finding the cause of autism.

## Data Gathering

Source: https://ida.loni.usc.edu/pages/access/search.jsp
Acquisition strategy: One-time download from website.
Database queries: Research group: "all"; Sex: "both", Modality: "fMRI" & "MRI", Image count: "ALL"
Privacy: From the website:
Consistent with the policies of the 1000 Functional Connectomes Project, data usage is unrestricted for non-commercial research purposes. We kindly request that the specific datasets included in analyses be specified appropriately, and that their funding sources be acknowledged. As per INDI protocol, we simply require that users register with the NITRC and 1000 Functional Connectomes Project to gain access. (Creative Commons, Attribution-NonCommercial-Share Alike License)

All data is anonymous and HIPPA compliant. 

## Data Assessment

Volume and coverage
- Is there enough data to support modeling?

Yes, the sample size in on par with other studies.
  
Granularity
- Does the level of detail match the unit of analysis?

Yes, we have the full demographic information and medical scans.

Bias and representativeness
- Consider missing subpopulations and selection bias.

Dataset only includes Western Europe and USA. The demographic information also does not include ethnicity. 

## Assessing Learnability

Signal vs. noise
- Do the features plausibly contain information about the target?

Yes, autism and control scans are labeled. 

Data sufficiency
- Are there enough examples overall and per class?

Yes, autism and control are roughly half and half. 

## KPI Definition 

- Primary KPI: accuracy
- Secondary KPI: fairness metrics -- accuracy based on age and/or gender
