import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm, AnovaRM
from statsmodels.sandbox.stats.multicomp import MultiComparison


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


def get_summary(df):
    if len(df["Condition"].unique()) > 1:
        stats = df.groupby(["Condition", "Feedback"])["acc"].agg(
            ["mean", "count", "std"]
        )
    else:
        stats = df.groupby(["Feedback"])["acc"].agg(["mean", "count", "std"])
    ci95_hi = []
    ci95_lo = []
    sem = []

    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.96 * s / math.sqrt(c))
        ci95_lo.append(m - 1.96 * s / math.sqrt(c))
        sem.append(s / math.sqrt(c))
    stats["sem"] = sem
    stats["ci95_hi"] = ci95_hi
    stats["ci95_lo"] = ci95_lo
    print(stats)
    return stats


def run_stats(df):
    if len(df["Condition"].unique()) > 1:
        formula = "acc ~ Condition + Feedback + Condition:Feedback"
    else:
        formula = "acc ~ Feedback"
    # we then run the model by calling formula using the data
    # in behav2 with the ordinary least squares method in
    # statsmodels (the ols function)
    model = ols(formula, df, missing="drop").fit()

    # then we create a table of the model results after
    # fitting the model
    aov_table = anova_lm(model, typ=2)
    aov_table

    # define a function to determine the eta squared value
    # eta squared gives the the proportion of variance
    # associated with one or more main treatments, errors or
    # interactions
    def eta_squared(aov):
        aov["eta_sq"] = "NaN"
        aov["eta_sq"] = aov[:-1]["sum_sq"] / sum(aov["sum_sq"])
        return aov

    # define a function to determine the omega squared value
    # omega squared is a measure of effect size, or the degree
    # of association for a population
    # omega squared estimates how much variance in the response
    # variables (in our case 'PercentImmobility') are accounted
    # for by the treatments
    # this is a lesser biased alternative to eta-squared,
    # especially when sample sizes are small
    def omega_squared(aov):
        mse = aov["sum_sq"][-1] / aov["df"][-1]
        aov["omega_sq"] = "NaN"
        aov["omega_sq"] = (aov[:-1]["sum_sq"] - (aov[:-1]["df"] * mse)) / (
            sum(aov["sum_sq"]) + mse
        )
        return aov

    eta_squared(aov_table)
    omega_squared(aov_table)

    # the eta_squared and omega_squared values get attached to
    # our aov_table
    print(aov_table.round(4))
    print(" - " * 10)

    # Tukey's HSD (Tukey's honest significant difference) for
    # determining which groups are significantly different
    # note that the second argument for the group identifying
    # factor only takes one value so if you have
    # multiple treatment factors in separate columns, combine
    # them to run the tukeyHSD on the resulting
    # mc_object

    # first we need to combine the Drugs and Hours columns to
    # get a total treatments column
    if len(df["Condition"].unique()) > 1:
        df["group"] = df["Feedback"].map(str) + " + " + df["Condition"].map(str)
        # now make the comparison object
        comparison1 = MultiComparison(df["acc"], df["group"])
        comparison2 = MultiComparison(df["acc"], df["Feedback"])
        comparison3 = MultiComparison(df["acc"], df["Condition"])
        results1 = pd.read_html(comparison1.tukeyhsd().summary().as_html())[0]
        results2 = pd.read_html(comparison2.tukeyhsd().summary().as_html())[0]
        results3 = pd.read_html(comparison3.tukeyhsd().summary().as_html())[0]
        results = pd.concat([results1, results2, results3])
    else:
        # now make the comparison object
        comparison1 = MultiComparison(df["acc"], df["Feedback"])
        results = pd.read_html(comparison1.tukeyhsd().summary().as_html())[0]
    # print(comparison.tukeyhsd())

    # print all the results where the p-value < 0.05
    # ps = results[results["reject"] == True]
    print(results)
    return results, aov_table.round(4)


def df_test_fov_img():
    d_real = pd.read_csv("results_test_fovimg_real_stim.csv", index_col=0)
    d_abstract = pd.read_csv("results_test_fovimg_abstract_stim.csv", index_col=0)

    for d in [d_real, d_abstract]:
        d.rename(
            columns={
                "net": "Feedback",
                "condition": "Condition",
            },
            inplace=True,
        )
    d_real.reset_index(drop=True, inplace=True)
    d_real = d_real.groupby(["Condition", "Feedback"]).apply(add_stats).reset_index()
    d_real.sort_values(["Feedback", "Condition"], inplace=True)
    d_real.loc[d_real["Feedback"] == "SiameseNet13", "Feedback"] = "V1 to V1"
    d_real.loc[d_real["Feedback"] == "SiameseNet23", "Feedback"] = "IT to V1"

    d_abstract.reset_index(drop=True, inplace=True)
    d_abstract = (
        d_abstract.groupby(["Condition", "Feedback"]).apply(add_stats).reset_index()
    )
    d_abstract.sort_values(["Feedback", "Condition"], inplace=True)
    d_abstract.loc[d_abstract["Feedback"] == "SiameseNet13", "Feedback"] = "V1 to V1"
    d_abstract.loc[d_abstract["Feedback"] == "SiameseNet23", "Feedback"] = "IT to V1"
    return d_real, d_abstract


def df_inspect_test_noise():
    d_real = pd.read_csv("results_test_noise_real_stim.csv", index_col=0)
    d_abstract = pd.read_csv("results_test_noise_abstract_stim.csv", index_col=0)

    for d in [d_real, d_abstract]:

        d.rename(
            columns={
                "net": "Feedback",
            },
            inplace=True,
        )
    d_real.loc[d_real["Feedback"] == "SiameseNet13", "Feedback"] = "V1 to V1"
    d_real.loc[d_real["Feedback"] == "SiameseNet23", "Feedback"] = "IT to V1"
    d_real["Condition"] = "Real"
    d_real = (
        d_real.groupby(["Condition", "Feedback", "noise_sd"])
        .apply(add_stats)
        .reset_index()
    )

    d_abstract.loc[d["Feedback"] == "SiameseNet13", "Feedback"] = "V1 to V1"
    d_abstract.loc[d["Feedback"] == "SiameseNet23", "Feedback"] = "IT to V1"
    d_abstract["Condition"] = "Abstract"
    d_abstract = (
        d_abstract.groupby(["Condition", "Feedback", "noise_sd"])
        .apply(add_stats)
        .reset_index()
    )

    return d_real, d_abstract


def df_inspect_test_classify():
    d_real = pd.read_csv("results_test_classify_real_stim.csv")
    d_real["condition"] = "real"
    d_real.sort_values("net", inplace=True)

    d_abstract = pd.read_csv("results_test_classify_abstract_stim.csv")
    d_abstract["condition"] = "abstract_stim"
    d_abstract.sort_values("net", inplace=True)

    for d in [d_real, d_abstract]:
        d.rename(
            columns={
                "net": "Feedback",
                "condition": "stim_type",
                "key": "Condition",
            },
            inplace=True,
        )
        d.loc[d["Feedback"] == "SiameseNet13", "Feedback"] = "V1 to V1"
        d.loc[d["Feedback"] == "SiameseNet23", "Feedback"] = "IT to V1"

        d.loc[d["Condition"] == "all", "Condition"] = "All"
        if "cb" in d["Condition"].unique():
            d.loc[d["Condition"] == "cb", "Condition"] = "Car vs. Bike"
            d.loc[d["Condition"] == "mf", "Condition"] = "Female vs. Male"
            d.loc[d["Condition"] == "fv", "Condition"] = "Face vs. Vehicle"
    return d_real, d_abstract


#############
pd.options.display.float_format = "{:,.3f}".format

# class_real, class_abs = df_inspect_test_classify()
# fov_real, fov_abs = df_test_fov_img()

# dfs = [class_real, class_abs]

# res_dict = {}

# for df in dfs:
#     varname = [
#         i
#         for i, a in locals().items()
#         if ((type(a) == pd.core.frame.DataFrame) and (a.equals(df)))
#     ][0]
#     print(f"#### {varname} ####")
#     summary = get_summary(df)
#     print(" - " * 10)
#     posthoc, anova = run_stats(df)
#     # print(stats.to_latex())
#     print("=" * 30)
#     res_dict[varname] = {
#         "summary": summary,
#         "anova": anova,
#         "posthoc": posthoc,
#     }

############## fov img
import pingouin as pg

fov_real, fov_abs = df_test_fov_img()

dfs = [fov_real, fov_abs]
dfs_names = ["fov_real", "fov_abs"]

res_dict = {"fov_img": {}, "noise": {}, "class": {}}
for i, fov in enumerate(dfs):
    # equal_var1 = pg.homoscedasticity(data=fov, dv="acc", group="Condition").equal_var[0]
    # equal_var2 = pg.homoscedasticity(data=fov, dv="acc", group="Feedback").equal_var[0]

    # anova_fov = pg.rm_anova(
    #     data=fov,
    #     dv="acc",
    #     subject="Feedback",
    #     within="Condition",
    #     effsize="n2",
    #     correction="bonf",
    #     detailed=True,
    # ).round(3)

    # ph_fov = pd.concat(
    #     [
    #         pg.pairwise_ttests(
    #             data=fov, dv="acc", between=["Condition", "Feedback"], padjust="bonf"
    #         ).round(3),
    #         pg.pairwise_ttests(
    #             data=fov, dv="acc", between=["Feedback", "Condition"], padjust="bonf"
    #         ).round(3)[-6:],
    #     ]
    # )

    anova_fov = pg.anova(
        data=fov,
        dv="acc",
        between=["Feedback", "Condition"],
        effsize="n2",
        detailed=True,
    ).round(3)

    ph_fov = pg.pairwise_ttests(
        data=fov,
        dv="acc",
        between=["Feedback", "Condition"],
        padjust="bonf",
    ).round(3)

    tab = ph_fov.drop(
        ["Paired", "Parametric", "alternative", "p-unc", "p-adjust"], axis=1
    )
    tab["dof"] = tab["dof"].astype(int)
    # print(anova_fov.to_latex())
    # print(tab.to_latex())

    res_dict["fov_img"][dfs_names[i]] = {
        "anova": anova_fov,
        "ph": ph_fov,
        "summary": get_summary(fov),
    }
########## class img
class_real, class_abs = df_inspect_test_classify()
dfs = [class_real, class_abs]
dfs_names = ["class_real", "class_abs"]

for i, fov in enumerate(dfs):
    # equal_var1 = pg.homoscedasticity(data=fov, dv="acc", group="Condition").equal_var[0]
    # equal_var2 = pg.homoscedasticity(data=fov, dv="acc", group="Feedback").equal_var[0]
    if "abs" in dfs_names[i]:
        anova_fov = pg.anova(
            data=fov,
            dv="acc",
            between="Feedback",
            effsize="n2",
            detailed=True,
        ).round(3)

        ph_fov = pg.pairwise_tukey(data=fov, dv="acc", between="Feedback").round(3)
        tab = ph_fov
    else:
        # anova_fov = pg.rm_anova(
        #     data=fov,
        #     dv="acc",
        #     subject="Feedback",
        #     within="Condition",
        #     effsize="n2",
        #     correction="bonf",
        #     detailed=True,
        # ).round(3)

        # ph_fov = pd.concat(
        #     [
        #         pg.pairwise_ttests(
        #             data=fov,
        #             dv="acc",
        #             between=["Condition", "Feedback"],
        #             padjust="bonf",
        #         ).round(3),
        #         pg.pairwise_ttests(
        #             data=fov,
        #             dv="acc",
        #             between=["Feedback", "Condition"],
        #             padjust="bonf",
        #         ).round(3)[7:],
        #     ]
        # )

        anova_fov = pg.anova(
            data=fov,
            dv="acc",
            between=["Feedback", "Condition"],
            effsize="n2",
            detailed=True,
        ).round(3)

        ph_fov = pd.concat(
            [
                pg.pairwise_ttests(
                    data=fov,
                    dv="acc",
                    between=["Condition", "Feedback"],
                    padjust="bonf",
                ).round(3),
                pg.pairwise_ttests(
                    data=fov,
                    dv="acc",
                    between=["Feedback", "Condition"],
                    padjust="bonf",
                ).round(3)[7:],
            ]
        )
        tab = ph_fov.drop(
            ["Paired", "Parametric", "alternative", "p-unc", "p-adjust"], axis=1
        )
        tab["dof"] = tab["dof"].astype(int)
    # print(anova_fov.to_latex())
    # print(tab.to_latex())

    res_dict["class"][dfs_names[i]] = {
        "anova": anova_fov,
        "ph": ph_fov,
        "summary": get_summary(fov),
    }
######### noise ttest
noise_real, noise_abs = df_inspect_test_noise()

ys_IT_abs = noise_abs.loc[
    (
        (noise_abs["Feedback"] == "IT to V1")
        & (
            (noise_abs["noise_sd"] == (0.0))
            | (round(noise_abs["noise_sd"]) == (49))
            | (noise_abs["noise_sd"] == (100.0))
        )
    )
]

ys_V1_abs = noise_abs.loc[
    (
        (noise_abs["Feedback"] == "V1 to V1")
        & (
            (noise_abs["noise_sd"] == (0.0))
            | (round(noise_abs["noise_sd"]) == (49.0))
            | (noise_abs["noise_sd"] == (100.0))
        )
    )
]
ys_IT_real = noise_real.loc[
    (
        (noise_real["Feedback"] == "IT to V1")
        & (
            (noise_real["noise_sd"] == (0.0))
            | (round(noise_real["noise_sd"]) == (49.0))
            | (noise_real["noise_sd"] == (100.0))
        )
    )
]
ys_V1_real = noise_real.loc[
    (
        (noise_real["Feedback"] == "V1 to V1")
        & (
            (noise_real["noise_sd"] == (0.0))
            | (round(noise_real["noise_sd"]) == (49.0))
            | (noise_real["noise_sd"] == (100.0))
        )
    )
]

yss = pd.concat([ys_IT_abs, ys_V1_abs, ys_IT_real, ys_V1_real])
# names = ["IT_abstract", "IT_real", "V1_abstract", "V1_real"]
noise0 = yss.loc[yss["noise_sd"] == 0.0]
noise49 = yss.loc[round(yss["noise_sd"]) == 49.0]
noise100 = yss.loc[yss["noise_sd"] == 100.0]

dfs = [noise0, noise49, noise100]
names = ["noise0", "noise49", "noise100"]

noises = pd.concat(dfs)
noise_anova = pg.anova(
    data=noises,
    dv="acc",
    between=["noise_sd"],
    effsize="n2",
    detailed=True,
).round(3)

noise_ph = pg.pairwise_ttests(
    data=noises, dv="acc", between=["noise_sd"], padjust="bonf"
).round(3)


for i, fov in enumerate(dfs):
    anova_fov = pg.anova(
        data=fov,
        dv="acc",
        between=["Condition", "Feedback"],
        effsize="n2",
        detailed=True,
    ).round(3)

    ph_fov = pd.concat(
        [
            pg.pairwise_ttests(
                data=fov, dv="acc", between=["Condition", "Feedback"], padjust="bonf"
            ).round(3),
            pg.pairwise_ttests(
                data=fov, dv="acc", between=["Feedback", "Condition"], padjust="bonf"
            ).round(3)[-3:],
        ]
    )

    res_dict["noise"][names[i]] = {
        "anova": anova_fov,
        "ph": ph_fov,
        "summary": get_summary(fov),
    }
# ####### noise curve fitting
# from scipy.optimize import curve_fit


# def monoExp(x, m, t, b):
#     return m * np.exp(-t * x) + b


# noise_real, noise_abs = df_inspect_test_noise()

# xs = noise_abs["noise_sd"].unique()

# ys_IT_abs = (
#     noise_abs.loc[noise_abs["Feedback"] == "IT to V1"]
#     .groupby(["Feedback", "noise_sd"])
#     .mean()["acc"]
#     .values
# )
# ys_V1_abs = (
#     noise_abs.loc[noise_abs["Feedback"] == "V1 to V1"]
#     .groupby(["Feedback", "noise_sd"])
#     .mean()["acc"]
#     .values
# )
# ys_IT_real = (
#     noise_real.loc[noise_real["Feedback"] == "IT to V1"]
#     .groupby(["Feedback", "noise_sd"])
#     .mean()["acc"]
#     .values
# )
# ys_V1_real = (
#     noise_real.loc[noise_real["Feedback"] == "V1 to V1"]
#     .groupby(["Feedback", "noise_sd"])
#     .mean()["acc"]
#     .values
# )

# yss = [ys_IT_abs, ys_V1_abs, ys_IT_real, ys_V1_real]
# names = ["IT_abstract", "IT_real", "V1_abstract", "V1_real"]

# from scipy.stats import ks_2samp

# # perform Kolmogorov-Smirnov test
# ks_s, ks_p = ks_2samp(ys_IT_real, ys_V1_real)
# print(ks_p < 0.05)


# for i, ys in enumerate(yss):
#     varname = names[i]
#     print(f"#### {varname} ####")

#     # perform the fit
#     p0 = (1, 0.1, 50)  # start with values near those we expect
#     params, cv = curve_fit(monoExp, xs, ys, p0=None, maxfev=5000)
#     m, t, b = params
#     sampleRate = 20_000  # Hz
#     tauSec = (1 / t) / sampleRate

#     # determine quality of the fit
#     squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
#     squaredDiffsFromMean = np.square(ys - np.mean(ys))
#     rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
#     print(f"R² = {rSquared}")

#     # plot the results
#     plt.plot(xs, ys, ".", label="data")
#     plt.plot(xs, monoExp(xs, m, t, b), "--", label="fitted")
#     plt.title("Fitted Exponential Curve")

#     # inspect the parameters
#     print(f"Y = {m} * e^(-{t} * x) + {b}")
#     print(f"Tau = {tauSec * 1e6} µs")

#     res_dict["noise"][varname] = {
#         "m": m,
#         "t": t,
#         "b": b,
#         "cv": cv,
#         "tauSec": tauSec,
#         "rSquared": rSquared,
#     }
# from scipy import stats
# import itertools

# combs = list(itertools.combinations(names, 2))

# for comb in combs:
#     print(comb)
#     x1 = [
#         res_dict["noise"][comb[0]]["m"],
#         res_dict["noise"][comb[0]]["t"],
#         res_dict["noise"][comb[0]]["b"],
#     ]
#     x2 = [
#         res_dict["noise"][comb[1]]["m"],
#         res_dict["noise"][comb[1]]["t"],
#         res_dict["noise"][comb[1]]["b"],
#     ]
#     tStat, pValue = stats.ttest_ind(
#         x1, x2, equal_var=False
#     )  # run independent sample T-Test
#     print("P-Value:{0} T-Statistic:{1}".format(pValue, tStat))
#     print(pValue < 0.05)
# #############
