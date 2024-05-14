# HeartBeat Sound Classifier

Semester 4 MLPR (Machine Learning and Pattern Recognition) Project to classify Heartbeat sounds intonormal/abnormal and detect murmurs in the sounds. We plan to use the [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/) dataset for this project. Course Details and more information on the project can be found on the [course website](https://ai3011.plaksha.edu.in/).

Mid-Semester Presentation: [here](https://www.canva.com/design/DAF-2KTqmMQ/1pH0lFmq6Pz7TaNhMOAjaw/edit?utm_content=DAF-2KTqmMQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
End-Semester Presentation: [here](https://www.canva.com/design/DAGEqXNYXT8/dbGEHCdjWGkWOFkQSwNlbQ/view?utm_content=DAGEqXNYXT8&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## Abstract (as on course website)
World Heart Federation (WHF) estimated that 20.5 million people died from cardiovascular diseases in 2023, representing 34% of all global deaths. Out of this, 72.1% were unaware of their hypertension status. Heart sound analysis is a non-invasive technique that can be used to facilitate early identification of cardiac abnormalities such as murmurs.

Our project aims to employ machine learning for the analysis of heart sounds collected via devices such as a digital stethoscope, to facilitate the early identification of cardiac abnormalities and murmurs. We aim to expand this feature to smartphones, where people can easily check their heart health and determine when medical advice is necessary.
We used the publicly available PhysioNet 2022 challenge dataset, which contains 5000+ heart sound recordings, labelled for murmur presence and normality.
Previous studies, such as McDonald et al. (2022) - the winners of the aforementioned challenge, use various classification techniques including CNNs and RNNs. None, however, apply the classification to classify Mobile phone heart recordings, or modify the model to account for the same.

After preprocessing the data by clipping, downsampling, and filtering using a butterworth filter, we used the extracted MFCC images of the sound as input to our model. 
After trying both CNNs and RNN (LSTM) for classification, we observed better performance with the CNN model. This model accounted for the temporal variation in the sound, and was robust enough to classify the sub-par mobile recordings.

Performance metrics - Cost and Weighted Accuracy, defined by the dataset creators Physionet, were used to evaluate our model. These were designed primarily to prevent false negatives (non-diagnosis of CVD). Our model would rank 1st amongst all challenge participants with a low cost of 8388, and a weighted accuracy of 0.716.


## Folders and Files explained
1. imgs/ : Contains images used in the Midsem presentation slides (mostly plots).
2. src/ : Contains the code for the project. This is the main folder where all the code will be written. It is divided into sensible subfolders for better organization, names of files and subfolders should be self-explanatory, but ask me if any doubts.
Most files have descriptions at the top, explaining what the code does.
Models have .md file separately explaining the model and the code.
3. data/ : Contains the data files that we have generated through preprocessing. The files are as follows:
    means_csv: Basic Models Data
    - full_data.csv : This is all the raw data recieved through preprocessing.
    - filtered_data.csv : This is the data after filtering out the unwanted data columns, through the correlation matrix method.
    spectograms: h5 files for spectograms for Deep Neural Network Models
4. docs/ : Random notes and documents that we have created during the project.

## Steps to setup the project
1. Clone the repository to your local machine, in a suitable location.
```bash
git clone https://github.com/coolperson111/heartbeat-sound-classifier.git
cd heartbeat-sound-classifier
```
2. Create a new directory called `datasets` in the root of the project directory.
3. Download the dataset from the [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/) website. Scroll to the bottom of the page to find the dataset.
4. Extract the contents of the dataset to the `datasets` directory. The exact path of the physionet dataset should be as follows:
```
datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/
```
5. Install the requirements
PS: I am using python3.11.2 for this, if any of the code is not running, it might be due to mismatched python versions.
```bash
pip install -r requirements.txt
```
