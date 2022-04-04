# Speaker Recognition


## Getting Started
Στο παρακάτω αρχείο είναι τα βήματα που ακολουθώ για να τρέξω το sidekit με μοντέλο τα i-vectors.

### Executing program

* 1. Κατεβάζω το sidekit στο colab απο το github και κατεβάζω κάποιες βιβλιοθήκες
```
%%capture
#Local installation
#!git clone https://github.com/943274923/Speaker-Recognition 
#!git clone https://git-lium.univ-lemans.fr/Larcher/sidekit.git 
!git clone https://github.com/Anwarvic/Speaker-Recognition.git
%cd /content/Speaker-Recognition
!pip install -r requirements.txt

```
* 2.  Κατεβάζω το sidekit στο colab απο το github και κατεβάζω κάποιες βιβλιοθήκες

```
!pip install sox
!pip install -r requirements.txt
!pip3 install torch
!pip3 install torchvision
!sudo apt-get install python3.6-tk
!sudo apt-get install libsvm-dev
!apt-get install libsox-fmt-all libsox-dev sox > /dev/null
!python -m pip install torchaudio > /dev/null
!python -m pip install git+https://github.com/facebookresearch/WavAugment.git > /dev/null

```

* 3. Τρέχω το data_init.py
```
!cd /content/Speaker-Recognition/
!python data_init.py
```

* 4. Τρέχω το extract_features.py

```
!cd /content/Speaker-Recognition/
!python extract_features.py
```



* 5. Αλλάζω την έκδοση του h5py για να μπορούν να τρέξουν κάποιες βιβλιοθήκες

```
!pip install libsvm
!pip install --upgrade pip && pip install h5py=='2.8.0'
```


* 6. Τρέχω το i-vector.py


```
!cd /content/Speaker-Recognition/
!python i-vector.py

```
