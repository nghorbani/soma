## SOMA Configuration
SOMA code uses [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) to control different settings 
while separating code from configuration files. 
Furthermore, OmegaConf YAML files are "smart" in the way that one can set dynamic value resolvers.
This means that a value for a config key can be dependent on other keys that are *lazily* determined during runtime. 
Refer to the OmegaConf tutorials for a full overview of capabilities.
This technique is used overall in this project including SOMA, MoSh++, evaluation code, and Blender rendering capabilities. 
You can find the configuration YAML files at 
```` soma/support_data/conf ````
MoSH++ code has a similar configuration in its _support_data_ directory. 
The entries in these files are default values that can be changed at runtime. 
The code in this repository does not provide a command-line interface. 
Instead, one can use Jupyter notebooks or plain Python scripts.

## Definitions
In an optical marker-based mocap system we place light-reflecting or -emitting markers on the subject's body. 
If markers are reflective then the system is called a **passive mocap system**; e.g. VICON. 
If markers are emitting light then it is called an **active mocap system**; i.e. PhaseSpace.

MoCap systems use a multi-camera setup to reconstruct **sparse 3D mocap point clouds** from the light obtained from the markers.
The correspondence of the 3D points and the markers on the body is called a **label**. 
Labeling the sparse mocap point cloud data is a bottleneck in capturing high-quality mocap data and SOMA addresses this issue.

A mocap point cloud has extra information compared to the usual point cloud in that it usually also has a small **tracklet** information.
That is the mocap hardware outputs groups of points that correspond to the same marker through time. 
SOMA assigns the most frequent predicted label of the group of points, and this we call as **tracklet labeling**.
Tracklet labeling enhances the stability of the labels.

Active mocap systems provide the full trajectory of the point since the emitted light has a unique coding frequency for each marker.
However, in archival data often markers are placed at arbitrary, undocumented locations, 
hence the correspondence of points to locations on the body, i.e. labeling, is still required.

**Ghost points** frequently happen due to mocap hardware hallucinations. Note that exist no ghost marker but ghost points.

**Marker occlusions** happen when not enough cameras see a marker in the scene 
and no point is reconstructed by mocap hardware for that marker.

**Markerlayout** defines a markerset and a specific placement of it on the body. 

**Markerset** is the set of markers used for a capture session. Markerset has no information about the marker placement. 

**Marker placement** is the actual implementation of a markerset on the subject body during the capture session.

Both the makerset and the marker placement are subject to change for a capture session. 
When the trial coordinator chooses to remove a marker or place a new one then the markerset is altered.
In the worst-case scenario, markers drop from the subject's body due to sweating, rapid motion, or physical shock and this also changes the markerset.
Trial coordinators place the markers by hand on the subject's body so marker placement is always subject to change. 
This can even happen during the capture when the mocap suit simply stretches or folds.

During the SOMA training, we provide only the significant variations of the markerlayout and 
one model is then capable of handling variations of the markerlayout throughout the dataset and even during the capture session.

In this repository, we use MoSh and MoSh++ interchangeably and always mean MoSh++ by it. Otherwise, we would mention it.

## Folder Structure
In all the tutorials we assume you have prepared the following directory structure. 
Apart from the directory for the code please create a root directory for holding training data and experiments,
runtime results, and support data. This directory will have the following sub-folders:

![alt text](https://download.is.tue.mpg.de/soma/tutorials/soma_main_folder.png)

For your convinience we have prepared a 
[template folder structure for SOMA](https://download.is.tue.mpg.de/soma/tutorials/SOMA_FOLDER_TEMPLATE.tar.bz2).

This structure is an example case for a model trained for [SOMA dataset](https://soma.is.tue.mpg.de/download.php); i.e. V48_02_SOMA.
Note that experiments can share one data ID and that is why the data ID, V48_01_SOMA, is different compared to the experiment ID.

Please obtain the SMPL-X body model with a locked head for SOMA from [this link](https://smpl-x.is.tue.mpg.de/download.php) and the
[extra smplx data](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smplx/extra_smplx_data.tar.bz2) 
and place them in the smplx folder as you see in the above image.


We assume a mocap session can have three identifiers: name of the project; i.e. dataset name, a subject name, and sequence names.
SOMA expects to receive mocap sequences in this folder structure: dataset>subject>sequence.c3d.
This is also the case for MoSh++. Additionally, MoSh++ expects to get a settings.json file inside the subject's folder 
that holds the subject gender information.


## Training SOMA

SOMA uses synthetic data for training. The main component of the data generation pipeline is the AMASS bodies in 
[SMPLx gender neutral format](https://amass.is.tue.mpg.de/download.php).
Originally SOMA is trained using parameters of
ACCAD, CMU, HumanEVA, PosePrior, Total Capture, and Transitions datasets. 
Download **SMPL-X gender neutral** _body data_ of these datasets and place them under
```` support_files/smplx/amass_neutral ````.

In addition, we use 3764 [CAESAR](https://www.humanics-es.com/CAESARvol1.pdf) SMPL-X parameters for more subject variations.
Due to licensing restrictions, we cannot release beta parameters for CAESAR subjects.
You can download the already prepared
[body parameters without CAESAR subjects](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smplx/data/V48_01_HDM05_NoCAESAR.tar.bz2).

SOMA places virtual markers on these bodies following a set of given markerlayouts.
A markerlayout could be a labeled mocap frame as a c3d   or a json file. 
If a c3d file is given SOMA will automatically run MoSh++ to obtain the markerset and marker placement for the markerlayout.
You can obtain the markerlayouts for the experiments in the paper from [here](https://soma.is.tue.mpg.de/download.php). 

SOMA uses AMASS marker noise model to help generalize to mocap hardware differences.
This model copies the noise for each label from the real AMASS mocap markers. 
The noise is the difference between the MoSh++ simulated markers and the real markers of the mocap dataset.
Due to license restrictions, AMASS doesn't release real marker data of the subset datasets
hence to be able to use this noise model you need either a given AMASS noise model or the original mocap markers of the mentioned datasets.
We release AMASS noise model for experiments in the paper.
You can find the web address for these datasets on [AMASS download page](https://amass.is.tue.mpg.de/download.php). 
In case you have downloaded the original marker data you can run the script at
```` src/soma/amass/prepare_amass_smplx.py ````
to produce MoSh simulated markers necessary for the AMASS marker noise model.

SOMA is implemented in PyTorch and the training code benefits from the 
[PyTorch Lightning](https://www.pytorchlightning.ai/) (PTL) framework. 
PTL standardizes the training procedure and enables easy multi GPU training.

We provide Jupyter notebooks to demonstrate use cases of SOMA with hands-on examples:
- [Running SOMA on SOMA dataset](src/tutorials/runing_soma_on_soma_dataset.ipynb).
- [Solving an already labeled dataset](src/tutorials/solving_an_already_labeled_mocap.ipynb).
- [Label Priming](src/tutorials/label_priming.ipynb).
- [Running SOMA on SOMA dataset](src/tutorials/runing_soma_on_soma_dataset.ipynb).
- [Producing Synthetic MoCap for Evaluation](src/tutorials/produce_synthetic_mocap.ipynb)
