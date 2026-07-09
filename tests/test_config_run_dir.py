from cartoon_diffusion.cli.evaluate import build_parser as eval_parser
from cartoon_diffusion.cli.sample_grid import build_parser as sample_parser
from cartoon_diffusion.cli.train import build_parser as train_parser
from cartoon_diffusion.config import (
    apply_eval_run_dir,
    apply_sample_run_dir,
    apply_train_run_dir,
    parse_with_config,
)


def test_train_config_file_and_cli_override(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "\n".join(
            [
                "dataset_variant: 10k",
                "root: data/from-config",
                "steps: 10",
                "batch: 4",
                "run_dir: " + str(tmp_path / "run"),
            ]
        )
    )
    args = parse_with_config(train_parser(), ["--config", str(cfg), "--steps", "20"])
    assert args.root == "data/from-config"
    assert args.batch == 4
    assert args.steps == 20


def test_run_dir_defaults(tmp_path):
    train_args = train_parser().parse_args(["--run_dir", str(tmp_path / "train")])
    apply_train_run_dir(train_args)
    assert train_args.ckpt.endswith("checkpoints/latest.pt")
    assert (tmp_path / "train" / "config.yaml").exists()

    eval_args = eval_parser().parse_args(["--run_dir", str(tmp_path / "eval")])
    apply_eval_run_dir(eval_args)
    assert eval_args.ckpt.endswith("checkpoints/latest.pt")
    assert eval_args.clf_ckpt.endswith("classifiers/classifier.pt")
    assert eval_args.results_dir.endswith("results")
    assert (tmp_path / "eval" / "eval_config.yaml").exists()

    sample_args = sample_parser().parse_args(["--run_dir", str(tmp_path / "sample")])
    apply_sample_run_dir(sample_args)
    assert sample_args.ckpt.endswith("checkpoints/latest.pt")
    assert sample_args.out_dir.endswith("grids")
    assert (tmp_path / "sample" / "sample_config.yaml").exists()
