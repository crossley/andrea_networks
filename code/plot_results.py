# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:20:08 2022

@author: andre
"""
import os, glob
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sn.set_style("darkgrid")


def add_stats(x):
    labels = x["labels"].to_numpy()
    pred = x["pred"].to_numpy()

    accs = []
    for i in range(len(x)):
        a = np.fromstring(labels[i][1:-1], sep=" ")
        b = np.fromstring(pred[i][1:-1], sep=" ")
        accs.append(sum(1 for x, y in zip(a, b) if x == y) / float(len(a)))
    # acc = np.mean(accs)
    # sd = np.std(accs)

    # labels_rec = []
    # pred_rec = []
    # for i in range(len(labels)):
    #     for j in range(len(labels[i])):
    #         if labels[i][j].isnumeric():
    #             labels_rec.append(int(labels[i][j]))
    #             pred_rec.append(int(pred[i][j]))
    # labels_rec = labels[i]
    # pred_rec = pred[i]
    # acc = (np.array(labels_rec) == np.array(pred_rec)).astype(float)
    return pd.DataFrame({"acc": accs})


def inspect_test_fov_img():
    d_real = pd.read_csv("results_test_fovimg_real_stim.csv", index_col=0)
    d_abstract = pd.read_csv("results_test_fovimg_abstract_stim.csv", index_col=0)

    d_real.reset_index(drop=True, inplace=True)
    d_real = d_real.groupby(["condition", "net"]).apply(add_stats).reset_index()
    d_real.sort_values(["net", "condition"], inplace=True)
    d_real.loc[d_real["net"] == "SiameseNet13", "net"] = "V1 to V1"
    d_real.loc[d_real["net"] == "SiameseNet23", "net"] = "IT to V1"
    d_real["condition"] = d_real["condition"].str.title()
    d_real["acc"] = d_real["acc"]

    d_abstract.reset_index(drop=True, inplace=True)
    d_abstract = d_abstract.groupby(["condition", "net"]).apply(add_stats).reset_index()
    d_abstract.sort_values(["net", "condition"], inplace=True)
    d_abstract.loc[d_abstract["net"] == "SiameseNet13", "net"] = "V1 to V1"
    d_abstract.loc[d_abstract["net"] == "SiameseNet23", "net"] = "IT to V1"
    d_abstract["condition"] = d_abstract["condition"].str.title()
    d_abstract["acc"] = d_abstract["acc"]

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))

    sn.barplot(
        data=d_real,
        x="condition",
        y="acc",
        hue="net",
        ax=ax[0, 0],
        errwidth=1,
        capsize=0.1,
    )
    ax[0, 0].set_ylim(0.2, 1)
    ax[0, 0].legend(title="Model feedback")
    ax[0, 0].set_ylabel("Test Accuracy")
    ax[0, 0].set_xlabel("Condition")
    ax[0, 0].set_title("Real-world stimuli")

    sn.barplot(
        data=d_abstract,
        x="condition",
        y="acc",
        hue="net",
        ax=ax[0, 1],
        errwidth=1,
        capsize=0.1,
    )
    ax[0, 1].set_ylim(0.2, 1)
    ax[0, 1].legend(title="Model feedback")
    ax[0, 1].set_ylabel("Test Accuracy")
    ax[0, 1].set_xlabel("Condition")
    ax[0, 1].set_title("Abstract stimuli")

    fig.suptitle("Model performance (Foveal image)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    # plt.savefig("../figures/results_test_fovimg_real_and_abstract.pdf")
    plt.show()
    plt.close()


def inspect_test_noise():
    d_real = pd.read_csv("results_test_noise_real_stim.csv", index_col=0)
    d_abstract = pd.read_csv("results_test_noise_abstract_stim.csv", index_col=0)

    d_real.reset_index(drop=True, inplace=True)
    d_real["condition"] = "Real-world stimuli"

    d_abstract.reset_index(drop=True, inplace=True)
    d_abstract["condition"] = "Abstract stimuli"

    d = pd.concat((d_real, d_abstract))
    d = d.rename(columns={"net": "Model feedback", "condition": "Condition"})

    d.loc[d["Model feedback"] == "SiameseNet13", "Model feedback"] = "V1 to V1"
    d.loc[d["Model feedback"] == "SiameseNet23", "Model feedback"] = "IT to V1"

    d = (
        d.groupby(["Condition", "Model feedback", "noise_sd"])
        .apply(add_stats)
        .reset_index()
    )

    fig = sn.relplot(
        data=d,
        x="noise_sd",
        y="acc",
        kind="line",
        hue="Model feedback",
        style="Condition",
    )

    fig.ax.set_ylabel("Test Accuracy")
    fig.ax.set_xlabel("Noise (Standard Deviation)")

    plt.suptitle("Model performance (Foveal noise)", fontsize=13, fontweight="bold")

    # plt.savefig("../figures/results_test_noise_real_and_abstract.pdf")
    plt.show()
    plt.close()


def inspect_test_classify():
    d_real = pd.read_csv("results_test_classify_real_stim.csv")
    d_real["condition"] = "real"
    d_real.rename(columns={"key": "class"}, inplace=True)
    d_real.sort_values("net", inplace=True)

    d_abstract = pd.read_csv("results_test_classify_abstract_stim.csv")
    d_abstract["condition"] = "abstract_stim"
    d_abstract.rename(columns={"key": "class"}, inplace=True)
    d_abstract.sort_values("net", inplace=True)

    for d in [d_real, d_abstract]:
        d.rename(
            columns={
                "net": "Model feedback",
                "condition": "Condition",
                "class": "Contrast",
            },
            inplace=True,
        )
        d.loc[d["Model feedback"] == "SiameseNet13", "Model feedback"] = "V1 to V1"
        d.loc[d["Model feedback"] == "SiameseNet23", "Model feedback"] = "IT to V1"

        d.loc[d["Contrast"] == "all", "Contrast"] = "All"
        if "cb" in d["Contrast"].unique():
            d.loc[d["Contrast"] == "cb", "Contrast"] = "Car vs. Bike"
            d.loc[d["Contrast"] == "mf", "Contrast"] = "Female vs. Male"
            d.loc[d["Contrast"] == "fv", "Contrast"] = "Face vs. Vehicle"
    fig, ax = plt.subplots(
        1, 2, squeeze=False, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]}
    )
    sn.barplot(
        data=d_real,
        x="Model feedback",
        y="acc",
        hue="Contrast",
        ax=ax[0, 0],
        capsize=0.05,
        errwidth=1,
    )
    ax[0, 0].set_ylim(0.7, 1)
    ax[0, 0].set_xlabel("Network")
    ax[0, 0].set_ylabel("Classification Accuracy")
    ax[0, 0].set_title("Real-world stimuli")

    sn.barplot(
        data=d_abstract,
        x="Model feedback",
        y="acc",
        hue="Contrast",
        ax=ax[0, 1],
        capsize=0.16,
        errwidth=1,
    )
    ax[0, 1].set_ylim(0.7, 1)
    # ax[0, 1].legend().remove()
    ax[0, 1].set_xlabel("Network")
    ax[0, 1].set_ylabel("Classification Accuracy")
    ax[0, 1].set_title("Abstract stimuli")

    plt.suptitle(
        "SVM Classification performance, 5-fold",
        fontsize=13,
        fontweight="bold",
    )
    # plt.savefig("../figures/results_test_classify_real_and_abstract.pdf")
    plt.show()
    plt.close()


inspect_test_fov_img()
inspect_test_noise()
inspect_test_classify()
