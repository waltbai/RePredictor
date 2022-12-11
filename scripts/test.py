"""Test scripts."""
import argparse
import logging
import os

from repredictor.utils import load_config


def parse_args():
    """Parse arguments in command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/scpredictor.yml")
    parser.add_argument("--device", "-d", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    opt = parse_args()
    config = load_config(opt.config)
    model_type = config["model"]["type"]
    if model_type == "scpredictor":
        from repredictor.predictor.scpredictor import Predictor
    elif model_type == "pmi":
        from repredictor.predictor.pmi import Predictor
    elif model_type == "event-comp":
        from repredictor.predictor.event_comp import Predictor
    elif model_type == "repredictor":
        from repredictor.predictor.repredictor import Predictor
    elif model_type == "repredictor_verb":
        from repredictor.predictor.repredictor_verb import Predictor
    else:
        raise KeyError(f"Unknown model type '{model_type}'!")
    checkpoint_dir = os.path.join(
        config["work_dir"], "models", config["model"]["name"], "best")
    model = Predictor.from_checkpoint(checkpoint_dir, device=opt.device)
    model.validate(verbose=True)
    model.test()
