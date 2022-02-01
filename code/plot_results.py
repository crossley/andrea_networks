# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:32:31 2022

@author: andre
"""

import os, glob
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

sn.set_style("darkgrid")

root = r"C:\Users\andre\OneDrive - Macquarie University\GitHub\andrea_networks\code"
# res_files = glob.glob(os.path.join(root,'*.csv'))
# res_dict = {os.path.basename(file): pd.read_csv(file, index_col=0) for file in res_files}


def inspect_test_fov_img(path):
    d_real = pd.read_csv(os.path.join(path, "results_test_fovimg_real_stim.csv"))
    d_abstract = pd.read_csv(
        os.path.join(path, "results_test_fovimg_abstract_stim.csv")
    )

    fig, ax = plt.subplots(1, 2, squeeze=False)
    sn.barplot(data=d_real, x="condition", y="te_acc", hue="condition", ax=ax[0, 0])
    sn.barplot(data=d_abstract, x="condition", y="te_acc", hue="condition", ax=ax[0, 1])

    ax[0, 0].set_ylim(0, 100)
    ax[0, 0].legend().remove()
    ax[0, 0].set_ylabel("Test Accuracy")
    ax[0, 0].set_title("Real-world stimuli")

    ax[0, 1].set_ylim(0, 100)
    ax[0, 1].legend().remove()
    ax[0, 1].set_ylabel("Test Accuracy")
    ax[0, 1].set_title("Abstract stimuli")

    fig.suptitle("Model performance (Foveal image)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "new_fovimg_real_and_abstract.pdf"))
    plt.show()
    plt.close()


def inspect_test_noise(path):
    d_real = pd.read_csv(os.path.join(path, "results_test_noise_real_stim.csv"))
    d_abstract = pd.read_csv(os.path.join(path, "results_test_noise_abstract_stim.csv"))

    d_real["condition"] = "Real-world stimuli"
    d_abstract["condition"] = "Abstract stimuli"

    d = pd.concat((d_real, d_abstract))

    ax = sn.scatterplot(data=d, x="noise_sd", y="te_acc", hue="condition")

    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Noise (Standard Deviation)")
    plt.suptitle("Model performance (Foveal noise)", fontsize=13, fontweight="bold")

    plt.show()
    plt.savefig(os.path.join(path, "new_noise_real_and_abstract.pdf"))
    plt.close()


def inspect_test_classify(path):
    d_all = pd.read_csv(os.path.join(path, "results_test_classify_all.csv"))
    d_cb = pd.read_csv(os.path.join(path, "results_test_classify_cb.csv"))
    d_fv = pd.read_csv(os.path.join(path, "results_test_classify_fv.csv"))
    d_mf = pd.read_csv(os.path.join(path, "results_test_classify_mf.csv"))

    d_all["class"] = "All"
    d_cb["class"] = "Car vs. Bike"
    d_fv["class"] = "Face vs. Vehicle"
    d_mf["class"] = "Male vs. Female"

    # d_all['condition'] = 'real_stim'
    # d_cb['condition'] = 'real_stim'
    # d_fv['condition'] = 'real_stim'
    # d_mf['condition'] = 'real_stim'

    # d_abstract = pd.read_csv(os.path.join(path, 'results_test_classify_all_abstract_stim.csv'))
    # d_abstract['class'] = 'abstract'
    # d_abstract['condition'] = 'abstract_stim'

    d = pd.concat((d_all, d_cb, d_fv, d_mf))

    ax = sn.barplot(data=d, x="class", y="acc", capsize=0.2, errwidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Classification Accuracy")
    plt.suptitle("SVM Classification performance", fontsize=13, fontweight="bold")

    plt.savefig(os.path.join(path, "new_classify_real.pdf"))
    plt.show()
    plt.close()


inspect_test_fov_img(root)
inspect_test_noise(root)
inspect_test_classify(root)
