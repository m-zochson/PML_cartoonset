import json

from cartoon_diffusion.plotting import load_run, plot_results


def test_load_run_accepts_legacy_and_flat_schema(tmp_path):
    legacy = tmp_path / "legacy_fidelity.json"
    legacy.write_text(
        json.dumps(
            {
                "meta": {"attrs": ["eye_color"], "dataset_variant": "10k"},
                "results": [
                    {"weight": 0.0, "per_attribute": {"eye_color": 0.5}, "mean": 0.5}
                ],
            }
        )
    )
    flat = tmp_path / "flat_fidelity.json"
    flat.write_text(
        json.dumps(
            {
                "meta": {"attrs": ["eye_color"], "dataset_variant": "10k", "sampler": "ddpm"},
                "results": [{"weight": 0.0, "eye_color": 0.75, "mean": 0.75}],
            }
        )
    )

    _, _, legacy_rows = load_run(legacy)
    _, _, flat_rows = load_run(flat)
    assert legacy_rows[0]["eye_color"] == 0.5
    assert flat_rows[0]["eye_color"] == 0.75


def test_plot_results_writes_png_and_pdf(tmp_path):
    path = tmp_path / "run_fidelity.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "attrs": ["eye_color"],
                    "dataset_variant": "10k",
                    "sampler": "ddpm",
                    "timesteps": 4,
                    "n_samples": 2,
                },
                "results": [{"weight": 0.0, "eye_color": 0.75, "mean": 0.75}],
            }
        )
    )
    plot_results(results_dir=str(tmp_path), files=[str(path)], no_comparison=True)
    assert (tmp_path / "run_fidelity.png").exists()
    assert (tmp_path / "run_fidelity.pdf").exists()
