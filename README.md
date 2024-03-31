# BDA-A2
The repository contains a basic Text based search application implementation using MapReduce Technique. Approx 5.6GB csv dataset was preprocessed using Pandas and then utilised in the later MapReduce imp;ementation to determine Vector Space Model, as part of an assignment for the Fundamental of Big Data Analytics (DS2004) course.

---

# Dependencies
* Apche Hadoop
* Pandas
* Python

---

# Usage
* `preprocess.py` Contains the implementation of preprocessing on csv dataset and outputs a 'PreProcess.txt' file which contains the Article ID along with Article Text.
* `mapper.py` Contains the implementation of simply reading the 'PreProcess.txt' file and maps.
* `reducer.py` Contains the implementation of all the calculations e.g making vocabulary, calculating TF, IDF and Tf/IDF Weights, creating Vector Space Model of each Article.

---

# Contributors
This project exists thanks to my fellow group members:

### Saim Nadeem (i221884@nu.edu.pk)
### Shazer Nadeem (i222043@nu.edu.pk)