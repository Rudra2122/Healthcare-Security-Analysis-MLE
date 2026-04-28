# Introduction

Precision medicine is an emerging approach for disease treatment and prevention that delivers
personalized care to individual patients by considering their genetic makeups, medical histories,
environments, and lifestyles. Despite the rapid advancement of precision medicine and its
considerable promise, several underlying technological challenges remain unsolved. One such
challenge of great importance is the security and privacy of precision health–related data, such as
genomic data and electronic health records, which stifle collaboration and hamper the full potential
of machine-learning (ML) algorithms. To preserve data privacy while providing ML solutions, in our
article, [Briguglio, et. al.](https://arxiv.org/abs/2102.03412), we provide three contributions.
First, we propose a generic machine learning with encryption (MLE) framework, which we used to build
an ML model that predicts cancer from one of the most recent comprehensive genomics datasets in the
field. Second, our framework’s prediction accuracy is slightly higher than that of the most recent
studies conducted on the same dataset, yet it maintains the privacy of the patients’ genomic data.
Third, to facilitate the validation, reproduction, and extension of this work, we provide an
open-source repository that contains:

* the design and implementation of the MLE framework (folder
  [`SystemArchitecture`](./SystemArchitecture)). Please, read below for more information.
* all the ML experiments and code (folder [`ModelTraining`](./ModelTraining))
* the final predictive model deployed and the MLE framework, both deployed to a free cloud service
  [`https://mle.isot.ca`](https://mle.isot.ca)

# How to Run the Code

This section walks you through everything you need to get the project running on your machine, from installing Python all the way to seeing results. Just follow the steps in order and you should be good to go.

## Step 1: Install Python

If you are on a Mac, the easiest way is through Homebrew. Open your terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

Once that is done, check that it installed correctly:

```bash
python3 --version
# You should see something like Python 3.11.x
```


## Step 2: Get the Project Files

Unzip the project and navigate into the right folder:

```bash
cd ~/Downloads
unzip Healthcare-Security-Analysis-MLE-master.zip
cd Healthcare-Security-Analysis-MLE-master/ModelTraining
```


## Step 3: Set Up a Virtual Environment

A virtual environment keeps all the project dependencies separate from the rest of your system. This is good practice and avoids version conflicts.

```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, your terminal prompt will show `(venv)` at the start. You need to activate this every time you open a new terminal window before running any of the scripts.


## Step 4: Install the Required Libraries

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib joblib
```

If you are on an Apple Silicon Mac (M1, M2, or M3), pip will automatically download the right version for your chip. No extra configuration needed.


## Step 5: Understanding the Files

Here is a quick overview of what each file does so you know what you are working with:

| File | What it does |
|------|-------------|
| `feature_table.csv` | The main dataset. Around 7800 patients, each with about 5600 genomic features |
| `Y.csv` | The labels. Which of the 22 cancer types each patient has |
| `ceofs_true.csv` | Pre trained logistic regression coefficients from the original paper |
| `compareModels_og_modified.py` | The original paper experiment. Trains and compares multiple models |
| `comparePrecision.py` | A quick test to see how integer scaled weights affect accuracy |
| `compareModels_improvement.py` | Our improved version with fixed data leakage, class balancing, a stacking ensemble, and a throughput benchmark |
| `ModelComparison.txt` | Output file from the original experiment |
| `ModelComparison_submission.txt` | Output file from the improved experiment |



## Step 6: Run the Quick Precision Test

This script is fast, takes about 2 minutes, and tests how much accuracy changes when you round the model weights to different levels of precision. This is directly related to the paper's encryption idea.

```bash
python3 comparePrecision.py
```

Expected output:

```
Full Precision: 91.85% accuracy
10^3: accuracy stabilises
10^7: ranks fully agree
```

## Step 7: Run the Original Paper Experiment

This reproduces what the paper reported. It is the replication step.

```bash
python3 compareModels_og_modified.py
```

> **Heads up:** This one takes a long time. Anywhere from 6 to 24 hours depending on your machine, because it runs a full grid search across many combinations of classifiers, transformations, and feature selection methods. You do not need to sit and watch it. Just let it run in the background.

You can check on progress anytime by opening a second terminal and running:

```bash
tail -f ModelComparison.txt
```

When it finishes you should see something like this at the bottom of the output file:

```
Best Performing Model:
Accuracy: 0.7713
Logistic Regression is the winner with 3500 features removed using Mutual Information...
```


## Step 8: Run the Improved Experiment

This is our extended version that fixes several issues from the original paper and adds new contributions. It is the main deliverable of this project.

```bash
python3 compareModels_improvement.py
```

> **Runtime:** Around 60 to 80 minutes on an Apple M3. Much faster than the original.

Results print to the terminal as each part finishes. Everything is also saved to `ModelComparison_submission.txt` so you will not lose anything even if you close the window.

You can monitor it live with:

```bash
tail -f ModelComparison_submission.txt
```

When everything is done, the output will look like this:

```
============================================================
FINAL RESULTS
============================================================
Pipeline LR          Accuracy: 77.65%   Balanced Accuracy: 69.12%
Balanced LR          Accuracy: 73.93%   Balanced Accuracy: 73.62%
Stacking LR + SVM    Accuracy: 75.63%   Balanced Accuracy: 73.85%

PREDICTION SPEED
Batch size 1     Time: 0.24 ms   Samples/sec: 4132
Batch size 10    Time: 0.58 ms   Samples/sec: 17126
Batch size 100   Time: 1.99 ms   Samples/sec: 50225
Batch size 1000  Time: 19.63 ms  Samples/sec: 50949

Best balanced accuracy: 73.85%
```


## What We Improved Over the Original Paper

| Improvement | What We Changed | What It Achieved |
|-------------|----------------|-----------------|
| **A: Leak free Pipeline** | Moved feature selection inside cross validation so the test data is never seen during training | More honest and trustworthy accuracy numbers |
| **B: Class Balancing** | Added `class_weight='balanced'` so all 22 cancer types are treated equally | Balanced accuracy improved by 4.50% |
| **C: Stacking Ensemble** | Combined Logistic Regression and Linear SVM using a meta learner that decides which model to trust | Best balanced accuracy of 73.85% |
| **D: Throughput Benchmark** | Measured how fast the model can make predictions at different batch sizes | 50949 samples per second at batch size 1000 |


# System Architecture of MLE Framework

The server is meant to be deployed as a service, referred to as MLE.service, which is not exposed to the network. Instead, nginx (or something similar) should be used as a reverse proxy which manages incoming HTTP traffic and forwards the appropriate traffic to the MLE service. The nginx reverse proxy is also deployed as a service called nginx.service. Below is a summary for maintenance after deploying on a Ubuntu machine with nginx. Instructions for deployment can be found [here](https://docs.microsoft.com/en-us/aspnet/core/host-and-deploy/linux-nginx?view=aspnetcore-5.0).

## Running and Configuring the Services

You can stop or restart a service, or check its status using:

    sudo systemctl [restart|stop|status] [MLE|nginx]

The MLE service can be configured by editing `/etc/systemd/system/MLE.service`. After making edits, reload with:

    systemctl daemon-reload

An example .NET service configuration can be found [here](https://docs.microsoft.com/en-us/aspnet/core/host-and-deploy/linux-nginx?view=aspnetcore-5.0#create-the-service-file). The `ExecStart` line should read:

    ExecStart=/usr/bin/dotnet /var/www/MLE/CDTS_PROJECT.dll

The nginx rules for the MLE service can be configured by editing `/etc/nginx/sites-available/default`. Global nginx rules can be configured by editing `/etc/nginx/nginx.conf`. After making edits run:

    sudo nginx -t

This verifies the syntax of your configuration files. Then apply the changes with:

    sudo nginx -s reload

An example nginx configuration can be found [here](https://docs.microsoft.com/en-us/aspnet/core/host-and-deploy/linux-nginx?view=aspnetcore-5.0#configure-nginx).

## Redeploying After Code Changes

`/Server` contains the server side application. From the Server directory:

* Compile with:
        dotnet publish --configuration Release
* Copy the published folder into `/var/www/MLE`:
        sudo rm /var/www/MLE/ -r
        sudo cp ./bin/Release/netcoreapp3.1/publish/ /var/www/MLE/ -r
* Restart the service:
        sudo systemctl restart MLE.service

`/Client` contains the client application. From the Client directory:

* Clone the repo to a Windows machine with .NET 3.1 installed and compile with:
        dotnet publish --configuration Release -r win-x64 -p:PublishSingleFile=true --self-contained true
* The compiled file will be at `./Client/bin/release/netcoreapp3.1/win-x64/publish/CDTS_Project.exe`. Rename it to `MLE.txt` and copy it to both locations:
        cp /home/[username]/Healthcare-Security-Analysis/Client/bin/Release/netcoreapp3.1/win-x64/publish/CDTS_PROJECT.exe /home/[username]/Healthcare-Security-Analysis/Server/wwwroot/DownloadableFiles/MLE.txt
        cp /home/[username]/Healthcare-Security-Analysis/Client/bin/Release/netcoreapp3.1/win-x64/publish/CDTS_PROJECT.exe /var/www/MLE/wwwroot/DownloadableFiles/MLE.txt
* Restart the service:
        sudo systemctl restart MLE.service


# References

* Briguglio, W., Moghaddam, P., Yousef, W. A., Traore, I., & Mamun, M. (2021) "Machine Learning in Precision Medicine to Preserve Privacy via Encryption". [arXiv Preprint, arXiv:2102.03412](https://arxiv.org/abs/2102.03412).


# Citation

Please cite this work as:

```
@Article{Briguglio2021MachineLearningPrecisionMedicine-arxiv,
  author =       {William Briguglio and Parisa Moghaddam and Waleed A. Yousef and Issa Traore and
                  Mohammad Mamun},
  title =        {Machine Learning in Precision Medicine to Preserve Privacy via Encryption},
  journal =      {arXiv Preprint, arXiv:2102.03412},
  year =         2021,
  url =          {https://github.com/isotlaboratory/Healthcare-Security-Analysis-MLE},
  primaryclass = {cs.LG}
}
```
