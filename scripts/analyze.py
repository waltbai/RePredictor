"""Test scripts."""
import argparse
import logging
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy

from repredictor.utils import load_config


def parse_args():
    """Parse arguments in command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config1", "-c1", default="config/scpredictor.yml")
    parser.add_argument("--config2", "-c2", default="config/scpredictor.yml")
    parser.add_argument("--device", "-d", default="cuda:0")
    return parser.parse_args()


def draw_subplot(event, attn, ax):
    verb, role, arg_roles, arg_values, arg_concepts = event
    elements = [str((verb, role))]
    for r, v, c in zip(arg_roles, arg_values, arg_concepts):
        elements.append(str((r, v, c)))


def draw_attn_heatmap(event, attn, name):
    """Draw the attention heatmap."""
    verb, role, arg_roles, arg_values, arg_concepts = event
    elements = [str((verb, role))]
    for r, v, c in zip(arg_roles, arg_values, arg_concepts):
        elements.append(str((r, v, c)))
    # Draw heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(attn, cmap="YlGn", vmin=0., vmax=1.)
    cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    cbar.ax.set_ylabel("Attention weight", rotation=-90, va="bottom")
    ax.set_xticks(numpy.arange(attn.shape[1]), labels=elements)
    ax.set_yticks(numpy.arange(attn.shape[0]), labels=elements)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)
    # Draw texts
    # threshold = attn.max() / 2.
    threshold = 0.5
    for i in range(len(elements)):
        for j in range(len(elements)):
            color = "black" if attn[i, j] < threshold else "white"
            text = im.axes.text(j, i, f"{attn[i, j]:.2f}",
                                horizontalalignment="center",
                                verticalalignment="center",
                                color=color)
    fig.tight_layout()
    plt.savefig(f"/data/users/bl/imgs/{name}.png")
    plt.close()


def analyze_model(config_path):
    config = load_config(config_path)
    model_type = config["model"]["type"]
    if model_type == "repredictor":
        from repredictor.predictor.repredictor import Predictor
    elif model_type == "scpredictor":
        from repredictor.predictor.scpredictor import Predictor
    else:
        raise KeyError(f"Model '{model_type}' cannot be analyzed!")
    checkpoint_dir = os.path.join(
        config["work_dir"], "models", config["model"]["name"], "best")
    model = Predictor.from_checkpoint(checkpoint_dir, device=opt.device)
    results = model.analyze()
    return results


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    opt = parse_args()
    results1 = analyze_model(opt.config1)
    results2 = analyze_model(opt.config2)
    numpy.set_printoptions(precision=3)
    correct_idx1 = results1["predict_label"] == results1["true_label"]
    multi_arg_idx1 = (results1["num_args"] >= 5).any(axis=1)
    correct_idx2 = results2["predict_label"] == results2["true_label"]
    # Load dev set
    dev_path = os.path.join("/data/users/bl/data/resep_data/repredictor_data", "dev.pkl")
    with open(dev_path, "rb") as f:
        dev_set = pickle.load(f)
    dev2_path = os.path.join("/data/users/bl/data/resep_data/scpredictor_data", "dev.pkl")
    with open(dev2_path, "rb") as f:
        dev2_set = pickle.load(f)
    # Analyze
    # multi_arg_idx2 = (results2["num_args"] >= 5).any(axis=1)
    from pprint import pprint
    for i in range(len(correct_idx1)):
        if correct_idx1[i] and not correct_idx2[i] and multi_arg_idx1[i]:
            context, choices, target = dev_set[i]
            context2, choices2, target2 = dev2_set[i]
            print(i)
            for j in range(len(context)):
                print(context[j], context2[j])
            print()
            for j in range(len(choices)):
                print(choices[j], choices2[j])
            print()
            print(results1["predict_label"][i], results2["predict_label"][i], target)
            input()

    # for i in range(len(correct_idx)):
    #     if correct_idx[i] and multi_arg_idx[i]:
    #
    #         context_idx = results["num_args"][i] >= 5
    #         for j in range(len(context)):
    #             if context_idx[j]:
    #                 num_args = results["num_args"][i, j]
    #                 name = f"{i}_{j}"
    #                 draw_attn_heatmap(context[j], results["attn"][i, j, 0, :num_args+1, :num_args+1], f"{name}_0")
    #                 draw_attn_heatmap(context[j], results["attn"][i, j, 1, :num_args+1, :num_args+1], f"{name}_1")
