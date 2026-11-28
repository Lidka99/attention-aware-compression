"""Przelicz i wygeneruj tabele LaTeX oraz CSV artykułu z zapisów E035–E051.

Wejście: audytowane CSV/JSON w runs/, bez obrazów i wag. Ślad źródeł:
paper/README.md oraz WYNIKI.md odpowiednich eksperymentów; wyjście: paper/tables/.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RUNS = {
    exp: ROOT / "runs" / exp / run
    for exp, run in {
        "E035": "20261107T183831Z_8b0b14e0",
        "E036": "20261107T231027Z_27bac186",
        "E037": "20261108T031759Z_855c0b33",
        "E041": "20261116T040008Z_9e34f1e9",
        "E042": "20261121T212257Z_207ae03f",
        "E046": "20261122T045655Z_0b0c53b7",
        "E047": "20261122T053804Z_ae2dec73",
        "E048": "train_20261124T232822Z_89dfc621",
        "E049": "eval_20261124T234559Z_b4ca081b",
        "E050": "dev_20261125T231044Z_d44c66e4",
        "E051": "dev_20261126T213703Z_20c2a1d5",
    }.items()
}
PAIRS = {
    "resnet_cifar10": ("ResNet18", "CIFAR-10"),
    "resnet_cifar100": ("ResNet18", "CIFAR-100"),
    "resnet_tiny": ("ResNet18", "Tiny ImageNet"),
    "preact_cifar10": ("PreActResNet18", "CIFAR-10"),
    "preact_cifar100": ("PreActResNet18", "CIFAR-100"),
}
TAGS = ("b30", "b50", "b70")
SEEDS = (42, 43, 44, 45)
OFFSET_LABEL = "Frozen offsets"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def analysis(exp, name):
    return RUNS[exp] / "analysis" / name


def verify_sha(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9), (a, b)


def pair_rows_e035():
    checked = read_json(analysis("E035", "verification.json"))
    data = read_json(analysis("E035", "data_verification.json"))
    assert checked["status"] == data["status"] == "PASS"
    assert checked["identical_dev_test_model_states"]
    assert data["tiny_pixels_equal"] == 10000 and not data["cifar100_reserve_used"]
    plans = read_json(RUNS["E035"] / "frozen.json")["plans"]
    result = read_json(analysis("E035", "analysis.json"))
    rows = read_csv(analysis("E035", "per_seed.csv"))
    assert len(rows) == 720
    score = {(r["phase"], r["pair"], r["tag"], int(r["seed"]), r["profile"], r["mode"]):
             float(r["top1"]) for r in rows}
    assert len(rows) == len(score)
    controls = {(r["pair"], int(r["seed"])): 100 * float(r["top1"])
                for r in read_csv(analysis("E035", "controls.csv"))
                if r["mode"] == "CMA" and r["phase"] == "test"}
    contrasts = {(r["pair"], r["tag"]): r for r in result["contrasts"]
                 if r["phase"] == "test" and r["mode"] == "CMA"}
    cells = []
    for pair, (arch, dataset) in PAIRS.items():
        for tag in TAGS:
            u, t = ([100 * score["test", pair, tag, seed, p, "CMA"] for seed in SEEDS] for p in ("U", "T"))
            gain = [b - a for a, b in zip(u, t)]
            c = contrasts[pair, tag]
            p = plans[pair][tag]["params"]
            assert 999 * p["U"] <= 1000 * p["T"] <= 1000 * p["U"]
            assert c["positive"] == sum(g > 0 for g in gain) == 4
            for got, want in ((mean(gain), c["mean"]), (stdev(gain), c["sd"]),
                              (mean(u), c["absolute"]["U"]["mean"]),
                              (mean(t), c["absolute"]["T"]["mean"])):
                close(got, want)
            cells.append(dict(architecture=arch, dataset=dataset,
                              split="official validation" if pair == "resnet_tiny" else "official test",
                              nominal_parameter_reduction_pct=int(tag[1:]),
                              uniform_parameters=p["U"], tilt_parameters=p["T"],
                              uniform_top1_pct=mean(u), tilt_top1_pct=mean(t),
                              full_top1_pct=mean(controls[pair, seed] for seed in SEEDS),
                              tilt_minus_uniform_pp=mean(gain), paired_sd_pp=stdev(gain), positive_pairs=4,
                              **{f"seed_{seed}_gain_pp": g for seed, g in zip(SEEDS, gain)}))
    assert len(cells) == result["primary_pass_n"] == 15
    assert sum(c["nominal_parameter_reduction_pct"] >= 50 and c["tilt_minus_uniform_pp"] >= 2
               for c in cells) == result["magnitude_pass_n"] == 9
    return cells, plans, score


def cost_rows(plans35, score35):
    for exp in ("E036", "E037"):
        checked = read_json(analysis(exp, "verification.json"))
        assert checked["status"] == "passed"
        verify_sha(analysis(exp, "analysis.json"), checked["analysis_sha256"])
        verify_sha(RUNS[exp] / "frozen.json", checked["frozen_sha256"])
    plans37 = read_json(RUNS["E037"] / "frozen.json")["plans"]
    rows37 = {(r["pair"], r["tag"], int(r["seed"])): r
              for r in read_csv(analysis("E037", "per_seed.csv")) if r["mode"] == "CMA"}
    contrasts37 = {(r["pair"], r["tag"]): r
                   for r in read_json(analysis("E037", "analysis.json"))["allocation"]
                   if r["mode"] == "CMA" and r["contrast"] == "T-Up"}
    ratios = {(r["pair"], r["tag"], int(r["seed"]), int(r["batch"])): r
              for r in read_csv(analysis("E036", "ratios.csv"))}
    cells = []
    for pair, (arch, dataset) in PAIRS.items():
        for tag in TAGS:
            rows = [rows37[pair, tag, seed] for seed in SEEDS]
            for seed, row in zip(SEEDS, rows):
                for p in ("U", "T"):
                    close(float(row[p]), score35["dev", pair, tag, seed, p, "CMA"])
            scores = {p: [100 * float(r[p]) for r in rows] for p in ("U", "T", "Up")}
            gain = [up - t for up, t in zip(scores["Up"], scores["T"])]
            close(mean(gain), -contrasts37[pair, tag]["mean"])
            close(stdev(gain), contrasts37[pair, tag]["sd"])
            p35 = plans35[pair][tag]["params"]
            p37 = plans37[pair][tag]
            assert p37["T_params"] == p35["T"] and p37["Uprime_params"] > p35["T"]
            assert 98 * p37["T_mac"] <= 100 * p37["Uprime_mac"] <= 100 * p37["T_mac"]
            mac_ratio = p37["T_mac"] / p37["U_mac"]
            for seed in SEEDS:
                for batch in (1, 128):
                    close(mac_ratio, float(ratios[pair, tag, seed, batch]["mac_ratio"]))
            throughput = [float(ratios[pair, tag, s, 128]["throughput_ratio"]) for s in SEEDS]
            latency = [float(ratios[pair, tag, s, 1]["latency_ratio"]) for s in SEEDS]
            cells.append(dict(architecture=arch, dataset=dataset, split="development",
                              anchor_nominal_parameter_reduction_pct=int(tag[1:]),
                              uniform_parameters=p35["U"], tilt_parameters=p35["T"],
                              wide_uniform_parameters=p37["Uprime_params"],
                              uniform_mac_per_image=p37["U_mac"], tilt_mac_per_image=p37["T_mac"],
                              wide_uniform_mac_per_image=p37["Uprime_mac"],
                              uniform_top1_pct=mean(scores["U"]), tilt_top1_pct=mean(scores["T"]),
                              wide_uniform_top1_pct=mean(scores["Up"]),
                              wide_uniform_minus_tilt_pp=mean(gain), paired_sd_pp=stdev(gain),
                              wide_uniform_positive_pairs=sum(g > 0 for g in gain),
                              tilt_over_uniform_mac=mac_ratio,
                              wide_uniform_over_tilt_parameters=p37["Uprime_params"] / p35["T"],
                              wide_uniform_over_tilt_mac=p37["Uprime_mac"] / p37["T_mac"],
                              batch128_throughput_ratio_tilt_over_uniform_mean=mean(throughput),
                              batch1_latency_ratio_tilt_over_uniform_mean=mean(latency),
                              batch1_latency_ratio_tilt_over_uniform_sd=stdev(latency)))
    assert len(cells) == 15 and sum(c["wide_uniform_minus_tilt_pp"] > 0 for c in cells) == 10
    assert all(c["wide_uniform_top1_pct"] > c["uniform_top1_pct"] for c in cells)
    return cells


def dual_rows():
    exp = "E041"
    assert read_json(analysis(exp, "audit.json"))["passed"]
    checked = read_json(analysis(exp, "table_audit.json"))
    assert checked["passed"] and checked["contrasts_recomputed"] == 30
    for path, digest in checked["source_hashes"].items():
        verify_sha(ROOT / path, digest)
    raw = read_csv(analysis(exp, "per_seed.csv"))
    out = []
    for pair, expected in (("resnet_cifar10", 0.11250), ("resnet_cifar100", 2.10375)):
        arch, dataset = PAIRS[pair]
        cma = {int(r["seed"]): {} for r in raw if r["pair"] == pair}
        for r in raw:
            if r["pair"] == pair and r["mode"] == "CMA":
                cma[int(r["seed"])][int(r["control"])] = (float(r["T"]), float(r["C"]))
        control_ids = (101, 102, 103, 104)
        assert set(cma) == set(SEEDS) and all(tuple(sorted(c)) == control_ids for c in cma.values())
        tilt = [cma[s][101][0] for s in SEEDS]
        control_means = [mean(cma[s][i][1] for i in control_ids) for s in SEEDS]
        gain = [t - c for t, c in zip(tilt, control_means)]
        close(mean(gain), expected)
        best = max(control_ids, key=lambda i: mean(cma[s][i][1] for s in SEEDS))
        best_scores = [cma[s][best][1] for s in SEEDS]
        ratios = [read_json(RUNS[exp] / f"seed_42/{pair}/control_{i}.json")["plan"]["ratios"]
                  for i in control_ids]
        assert all(0.999 <= r <= 1 for pair_ratios in ratios for r in pair_ratios)
        out.append(dict(architecture=arch, dataset=dataset, split="development",
                        nominal_parameter_reduction_pct=50, tilt_top1_pct=mean(tilt),
                        mean_control_top1_pct=mean(control_means), tilt_minus_mean_control_pp=mean(gain),
                        paired_sd_pp=stdev(gain), positive_pairs=sum(g > 0 for g in gain),
                        best_control_id=best, best_control_top1_pct=mean(best_scores),
                        tilt_minus_best_control_pp=mean(tilt) - mean(best_scores),
                        best_control_positive_pairs=sum(c > t for t, c in zip(tilt, best_scores)),
                        min_control_over_tilt_parameters=min(r[0] for r in ratios),
                        max_control_over_tilt_parameters=max(r[0] for r in ratios),
                        min_control_over_tilt_mac=min(r[1] for r in ratios),
                        max_control_over_tilt_mac=max(r[1] for r in ratios)))
    return out


def b10_rows():
    results, paired_rows, contrasts = [], [], []
    models = {"teacher": "Full model", "b10": "Uniform folding", "L06": OFFSET_LABEL}
    for exp, test, n, profile_key, hits_key, n_key, hash_key in (
        ("E046", "Tiny ImageNet dev", 10000, "profile", "correct", "samples", "analyzed_sha256"),
        ("E047", "ImageNetV2", 2000, "model", "hits", "n", "output_sha256"),
    ):
        audit = read_json(analysis(exp, "audit.json"))
        assert audit["status"] == "PASS" and audit["assessments"] == 12
        for name in ("summary.json", "per_seed.csv"):
            verify_sha(analysis(exp, name), audit[hash_key][name])
        if exp == "E047":
            assert audit["exact_E046_models"] == 12
        summary = read_json(analysis(exp, "summary.json"))
        raw = read_csv(analysis(exp, "per_seed.csv"))
        assert all(int(r[n_key]) == n for r in raw)
        rows = {(p, s): next(r for r in raw if r[profile_key] == p and int(r["seed"]) == s)
                for p in models for s in SEEDS}
        assert len(rows) == len(raw) == 12
        hits = {k: int(r[hits_key]) for k, r in rows.items()}
        for p, label in models.items():
            m = summary["models"][p]
            costs = {key: int(m[key]) for key in ("params", "mac")}
            accuracies, losses = [], []
            for seed in SEEDS:
                r = rows[p, seed]
                assert all(int(r[k]) == v for k, v in costs.items())
                accuracy = 100 * hits[p, seed] / n
                loss = 100 * (hits["teacher", seed] - hits[p, seed]) / n
                close(accuracy, float(r["top1_pct"]))
                close(loss, float(r["loss_pp"]))
                accuracies.append(accuracy)
                losses.append(loss)
                paired_rows.append(dict(experiment=exp, sample=test, images=n, seed=seed, model_id=p, model=label,
                                        parameters=costs["params"], mac_per_image=costs["mac"],
                                        correct=hits[p, seed], top1_pct=accuracy, loss_to_full_pp=loss))
            for metric, vals in (("top1_pct", accuracies), ("loss_pp", losses)):
                close(mean(vals), m[metric]["mean"])
                close(stdev(vals), m[metric]["sd"])
            results.append(dict(experiment=exp, sample=test, images=n, model_id=p, model=label,
                                parameters=costs["params"], mac_per_image=costs["mac"], top1_pct=mean(accuracies),
                                top1_sd_pp=stdev(accuracies), paired_loss_to_full_pp=mean(losses),
                                paired_loss_sd_pp=stdev(losses)))
        gain = [100 * (hits["L06", s] - hits["b10", s]) / n for s in SEEDS]
        saved = summary["loss_uniform_minus_l06_pp"]
        assert gain == saved["per_seed"] and all(g > 0 for g in gain)
        close(mean(gain), saved["mean"])
        close(stdev(gain), saved["sd"])
        low, high = saved.get("ci95_pp", (None, None))
        contrasts.append(dict(experiment=exp, sample=test, images=n, l06_minus_uniform_pp=mean(gain),
                              paired_sd_pp=stdev(gain), positive_pairs=sum(g > 0 for g in gain),
                              image_bootstrap_ci95_low_pp=low, image_bootstrap_ci95_high_pp=high,
                              **{f"seed_{seed}_gain_pp": g for seed, g in zip(SEEDS, gain)}))
    for dev, external in zip(results[:3], results[3:]):
        assert all(dev[k] == external[k] for k in ("model_id", "model", "parameters", "mac_per_image"))
    for key in ("parameters", "mac_per_image"):
        assert 999 * results[1][key] <= 1000 * results[2][key] < 1000 * results[1][key]
    close(contrasts[0]["l06_minus_uniform_pp"], 1.1425)
    close(contrasts[1]["l06_minus_uniform_pp"], 1.3500)
    close(contrasts[1]["image_bootstrap_ci95_low_pp"], 0.700)
    close(contrasts[1]["image_bootstrap_ci95_high_pp"], 2.025)
    return results, paired_rows, contrasts


TRANSFER_DATASETS = {"cifar100": ("CIFAR-100", "official test", 5000),
                     "tiny": ("Tiny ImageNet", "official validation", 10000)}
TRANSFER_MODELS = {"full_pre": "Full model (native BN)", "full": "Full model", "b10_U": "UNIFORM",
                   "b10_Tproj": "T-proj", "b10_L06": OFFSET_LABEL}


def b10_transfer_rows():
    run = RUNS["E049"]
    manifest, audit = (read_json(run / name) for name in ("manifest.json", "audit.json"))
    baseline = read_json(RUNS["E048"] / "audit.json")
    assert manifest["status"] == "completed" and manifest["exit_codes"] == [0] * 4
    assert baseline["status"] == audit["status"] == "passed"
    assert baseline["validated_checkpoints"] == 8 and audit["baseline_audit"] == baseline
    assert audit["benchmark_previously_used"] and not audit["joint_gate_met"]
    verify_sha(run / "geometry.json", manifest["geometry_sha256"])
    verify_sha(run / "frozen.json", manifest["frozen_sha256"])
    geometry = read_json(run / "geometry.json")
    assert geometry == read_json(run / "frozen.json")["geometry"]
    raw = read_csv(run / "per_seed.csv")
    score = {(r["dataset"], r["split"], r["variant"], int(r["seed"])): r for r in raw}
    assert len(score) == len(raw) == 176
    results, per_seed, contrasts = [], [], []
    for dataset, (name, benchmark, n_dev) in TRANSFER_DATASETS.items():
        plans = geometry[dataset]["profiles"]
        outcome = audit["datasets"][dataset]
        assert outcome["E048_quality_gate_met"]
        for model in ("b10_U", "b10_Tproj", "b10_L06"):
            for key in ("parameters", "mac"):
                cap, observed = plans["b10_U"][key], plans[model][key]
                assert 999 * cap <= 1000 * observed <= 1000 * cap
                assert 1000 * observed <= 905 * plans["full"][key]
        for split, sample, n in (("benchmark", benchmark, 10000), ("dev", "development", n_dev)):
            counts = {p: [int(score[dataset, split, p, s]["correct"]) for s in SEEDS]
                      for p in TRANSFER_MODELS}
            for p, profile in TRANSFER_MODELS.items():
                plan = plans[p]
                losses = [100 * (full - acc) / n for full, acc in zip(counts["full"], counts[p])]
                top1 = 100 * mean(counts[p]) / n
                close(top1, outcome[split]["top1_mean_pct"][p])
                close(mean(losses), outcome[split]["loss_to_full_pp"][p])
                for s, hit, loss in zip(SEEDS, counts[p], losses):
                    row = score[dataset, split, p, s]
                    assert int(row["n"]) == n and 0 <= hit <= n
                    assert tuple(int(row[k]) for k in ("parameters", "mac")) == (plan["parameters"], plan["mac"])
                    assert row["state_sha256"] == score[dataset, "dev", p, s]["state_sha256"]
                    close(100 * hit / n, float(row["top1_pct"]))
                    per_seed.append(dict(dataset=name, sample=sample, images=n, model_id=p, model=profile,
                                         seed=s, correct=hit, top1_pct=100 * hit / n, loss_to_full_pp=loss,
                                         parameters=plan["parameters"], mac_per_image=plan["mac"]))
                if split == "benchmark" and p != "full_pre":
                    results.append(dict(dataset=name, sample=sample, images=n, model_id=p, model=profile,
                                        top1_pct=top1, loss_to_full_pp=mean(losses),
                                        parameters=plan["parameters"], mac_per_image=plan["mac"],
                                        parameter_reduction_pct=100 * (1 - plan["parameters"] / plans["full"]["parameters"]),
                                        mac_reduction_pct=100 * (1 - plan["mac"] / plans["full"]["mac"])))
            for p in ("b10_U", "b10_Tproj"):
                values = [100 * (a - b) / n for a, b in zip(counts["b10_L06"], counts[p])]
                saved = outcome[split]["contrasts"][f"b10_L06-{p}"]
                close(mean(values), saved["mean_pp"])
                close(stdev(values), saved["sd_pp"])
                for got, expected in zip(values, saved["per_seed_pp"], strict=True):
                    close(got, expected)
                positive = sum(v > 0 for v in values)
                assert positive == saved["positive_pairs"]
                contrasts.append(dict(dataset=name, sample=sample, images=n, comparison=f"b10_L06-{p}",
                                      mean_pp=mean(values), paired_sd_pp=stdev(values), positive_pairs=positive,
                                      **{f"seed_{s}_gain_pp": v for s, v in zip(SEEDS, values)}))
            if split == "benchmark":
                gain = [a - b for a, b in zip(counts["b10_L06"], counts["b10_U"])]
                loss = sum(counts["full"]) - sum(counts["b10_L06"])
                gate = (100 * sum(gain) >= len(SEEDS) * n and sum(v > 0 for v in gain) >= 3
                        and 100 * loss <= 3 * len(SEEDS) * n)
                assert not gate and gate == outcome["passed"]
            else:
                close(100 * mean(counts["full_pre"]) / n, baseline["results"][dataset]["mean_pct"])
    return results, per_seed, contrasts


def b10_crop_rows(transfer):
    run = RUNS["E050"]
    m, audit = (read_json(run / name) for name in ("manifest.json", "audit.json"))
    assert m["status"] == "completed" and m["exit_codes"] == [0] * 4 and m["dev_only"]
    assert audit["status"] == "passed" and not audit["pilot_gate_met"] and audit["reproduced_E049"] == 16
    raw = read_csv(run / "per_seed.csv")
    score = {(r["views"], r["profile"], int(r["seed"])): r for r in raw}
    control = {(r["model_id"], r["seed"]): r for r in transfer
               if r["dataset"] == "Tiny ImageNet" and r["sample"] == "development"}
    assert len(score) == len(raw) == audit["scores"] == 32
    results = []
    for p in ("full", "b10_U", "b10_Tproj", "b10_L06"):
        a, b = ([score[v, p, s] for s in SEEDS] for v in ("crop040", "crop065"))
        legacy, mild, change = [], [], []
        for s, old, new in zip(SEEDS, a, b):
            ref = control[p, s]
            assert old["parameters_sha256"] == new["parameters_sha256"]
            for row in (old, new):
                assert (int(row["n"]), int(row["parameters"]), int(row["mac"])) == (
                    ref["images"], ref["parameters"], ref["mac_per_image"])
            assert int(old["correct"]) == ref["correct"]
            legacy.append(int(old["correct"]) / 100)
            mild.append(int(new["correct"]) / 100)
            change.append((int(new["correct"]) - int(old["correct"])) / 100)
        close(mean(legacy), audit["recipes"]["crop040"]["top1_mean_pct"][p])
        close(mean(mild), audit["recipes"]["crop065"]["top1_mean_pct"][p])
        close(mean(change), audit["shift_in_top1_by_profile"][p]["mean_pp"])
        assert sum(v > 0 for v in change) == audit["shift_in_top1_by_profile"][p]["positive_pairs"] == 4
        results.append(dict(model_id=p, model=TRANSFER_MODELS[p], crop040_top1_pct=mean(legacy),
                            crop065_top1_pct=mean(mild), change_pp=mean(change), positive_pairs=4,
                            **{f"seed_{s}_change_pp": v for s, v in zip(SEEDS, change)}))
    return results


def tilt_new_rows():
    run = RUNS["E049"]
    audit = read_json(run / "audit.json")["datasets"]
    geometry = read_json(run / "geometry.json")
    rows = [r for r in read_csv(run / "per_seed.csv") if r["split"] == "benchmark"]
    score = {(r["dataset"], r["variant"], int(r["seed"])): r for r in rows}
    assert len(score) == len(rows) == 88
    out = []
    for ds, (dataset, sample, _) in TRANSFER_DATASETS.items():
        plans = geometry[ds]["profiles"]
        expected = audit[ds]["benchmark"]
        for tag in TAGS:
            u, t = tag + "_U", tag + "_T"
            costs = {p: (plans[p]["parameters"], plans[p]["mac"]) for p in ("full", u, t)}
            assert 999 * costs[u][0] <= 1000 * costs[t][0] <= 1000 * costs[u][0]
            hits = {}
            for variant, cost in costs.items():
                counts = []
                for seed in SEEDS:
                    r = score[ds, variant, seed]
                    assert tuple(int(r[k]) for k in ("parameters", "mac")) == cost
                    assert int(r["n"]) == 10000 and 0 <= int(r["correct"]) <= 10000
                    close(int(r["correct"]) / 100, float(r["top1_pct"]))
                    counts.append(int(r["correct"]))
                hits[variant] = counts
                close(mean(counts) / 100, expected["top1_mean_pct"][variant])
            loss = mean(a - b for a, b in zip(hits["full"], hits[t])) / 100
            close(loss, expected["loss_to_full_pp"][t])
            gain = [(a - b) / 100 for a, b in zip(hits[t], hits[u])]
            saved = expected["contrasts"][f"{t}-{u}"]
            for value, archived in zip(gain, saved["per_seed_pp"], strict=True):
                close(value, archived)
            close(mean(gain), saved["mean_pp"])
            close(stdev(gain), saved["sd_pp"])
            assert saved["positive_pairs"] == sum(g > 0 for g in gain) == 4
            out.append(dict(dataset=dataset, sample=sample, nominal_parameter_reduction_pct=int(tag[1:]),
                            full_top1_pct=mean(hits["full"]) / 100, uniform_top1_pct=mean(hits[u]) / 100,
                            tilt_top1_pct=mean(hits[t]) / 100, tilt_minus_uniform_pp=mean(gain),
                            paired_sd_pp=stdev(gain), positive_pairs=4, tilt_loss_to_full_pp=loss,
                            full_parameters=costs["full"][0], uniform_parameters=costs[u][0], tilt_parameters=costs[t][0],
                            full_mac_per_image=costs["full"][1], uniform_mac_per_image=costs[u][1], tilt_mac_per_image=costs[t][1],
                            tilt_over_uniform_mac=costs[t][1] / costs[u][1],
                            **{f"seed_{s}_gain_pp": value for s, value in zip(SEEDS, gain)}))
    assert len(out) == 6
    return out


def tilt_mild_rows():
    run = RUNS["E051"]
    m, audit = (read_json(run / name) for name in ("manifest.json", "audit.json"))
    assert m["status"] == "completed" and m["dev_only"] and m["exit_codes"] == [0] * 4
    assert audit["status"] == "passed" and audit["reproduced_controls"] == 32
    assert audit["dev_only"] and not audit["joint_b10_dev_gate_met"]
    verify_sha(run / "frozen.json", m["frozen_sha256"])
    geom = read_json(run / "frozen.json")["geometry"]
    raw = read_csv(run / "per_seed.csv")
    score = {(r["dataset"], r["variant"], int(r["seed"])): r for r in raw}
    assert len(score) == len(raw) == audit["scores"] == 56
    out = []
    for ds, (name, _, n) in TRANSFER_DATASETS.items():
        saved = audit["datasets"][ds]
        p0, mac0 = (geom[ds]["full"][k] for k in ("parameters", "mac"))
        for budget in (10, 5):
            u, t = f"b{budget}_U", f"b{budget}_T"
            p_u, m_u = (geom[ds][u][k] for k in ("parameters", "mac"))
            p_t, m_t = (geom[ds][t][k] for k in ("parameters", "mac"))
            assert 999 * p_u <= 1000 * p_t <= 1000 * p_u
            hits = {}
            for v in ("full", u, t):
                hits[v] = []
                for seed in SEEDS:
                    r = score[ds, v, seed]
                    assert int(r["n"]) == n and 0 <= int(r["correct"]) <= n
                    assert all(int(r[k]) == geom[ds][v][k] for k in ("parameters", "mac"))
                    hits[v].append(int(r["correct"]))
                    close(int(r["correct"]) * 100 / n, float(r["top1_pct"]))
                close(100 * mean(hits[v]) / n, saved["top1_mean_pct"][v])
            deltas = [100 * (a - b) / n for a, b in zip(hits[t], hits[u])]
            loss = 100 * (mean(hits["full"]) - mean(hits[t])) / n
            diff = saved["contrasts"][f"{t}-{u}"]
            for got, want in zip(deltas, diff["per_seed_pp"], strict=True):
                close(got, want)
            close(mean(deltas), diff["mean_pp"])
            close(stdev(deltas), diff["sample_sd_pp"])
            close(loss, saved["loss_to_full_pp"][t])
            assert sum(x > 0 for x in deltas) == diff["positive_pairs"] == 4
            gate = (1000 * p_t <= 905 * p0 and mean(deltas) >= 1 and loss <= 3) if budget == 10 else None
            if budget == 10:
                assert gate == saved["b10_dev_gate_met"] == (ds == "cifar100")
            out.append(dict(dataset=name, sample="development", images=n, nominal_parameter_reduction_pct=budget,
                            full_top1_pct=100 * mean(hits["full"]) / n, uniform_top1_pct=100 * mean(hits[u]) / n,
                            tilt_top1_pct=100 * mean(hits[t]) / n, tilt_minus_uniform_pp=mean(deltas),
                            tilt_loss_to_full_pp=loss, paired_sd_pp=stdev(deltas), positive_pairs=4, b10_dev_gate_met=gate,
                            full_parameters=p0, uniform_parameters=p_u, tilt_parameters=p_t,
                            full_mac_per_image=mac0, uniform_mac_per_image=m_u, tilt_mac_per_image=m_t,
                            tilt_over_uniform_mac=m_t / m_u, tilt_parameter_reduction_pct=100 * (1 - p_t / p0),
                            tilt_mac_reduction_pct=100 * (1 - m_t / mac0),
                            **{f"seed_{seed}_gain_pp": value for seed, value in zip(SEEDS, deltas)}))
    return out


def report(e035, cost, dual, b10, contrasts):
    print(f"E035 CMA: {len(e035)}/15 configurations, all 4/4 paired gains.")
    for key in ("tilt_over_uniform_mac", "wide_uniform_over_tilt_parameters", "wide_uniform_over_tilt_mac",
                "batch128_throughput_ratio_tilt_over_uniform_mean", "batch1_latency_ratio_tilt_over_uniform_mean",
                "batch1_latency_ratio_tilt_over_uniform_sd"):
        values = [r[key] for r in cost]
        print(f"{key}: {min(values):.6f} to {max(values):.6f}")
    print("E037: wider UNIFORM wins 10/15 on development data.")
    for r in dual:
        print(f"E041 {r['dataset']}: {r['tilt_minus_mean_control_pp']:+.5f} pp, {r['positive_pairs']}/4.")
    candidates = [r for r in read_csv(analysis("E042", "profiles.csv"))
                  if r["pair"] == "resnet_cifar100" and r["profile"] not in ("U", "T-proj")]
    assert read_json(analysis("E042", "audit.json"))["passed"] and len(candidates) == 24
    assert max(candidates, key=lambda r: float(r["top1"]))["profile"] == "L06"
    effects = [float(r["vs_U_pp"]) for r in candidates]
    assert sum(v > 0 for v in effects) == 5
    close(median(effects), -1.265)
    print("E042: L06 selected on CIFAR-100 dev; 5/24 candidates improved on UNIFORM.")
    for c in contrasts:
        print(f"{c['experiment']} {c['sample']}: {c['l06_minus_uniform_pp']:+.4f} pp, {c['positive_pairs']}/4.")
    print("E047: audited 95% paired image bootstrap [+0.700, +2.025] pp.")
    for key in ("parameters", "mac_per_image"):
        print(f"b10 L06 reduction vs full ({key}): {100 * (1 - b10[2][key] / b10[0][key]):.6f}%")


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: f"{v:.12g}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    return stream.getvalue()


def tex_row(*items):
    return " & ".join(items) + r" \\"


def tex_table(caption, label, columns, headings, rows, spacing=5):
    return "\n".join([
        "% Generated from the audited runs by paper/prepare_tables.py.",
        r"\begin{table}[tbp]", r"\centering", r"\small",
        rf"\setlength{{\tabcolsep}}{{{spacing}pt}}",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        *headings, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def grouped_rows(rows):
    previous = None
    for r in rows:
        key = r["architecture"], r["dataset"]
        yield key != previous, previous is None, r
        previous = key


def tex_e035(rows):
    body = []
    for new, first, r in grouped_rows(rows):
        if new and not first:
            body.append(r"\addlinespace[2pt]")
        body.append(tex_row(
            r["architecture"] if new else "", r["dataset"] if new else "",
            rf"{r['nominal_parameter_reduction_pct']}\%", f"{r['full_top1_pct']:.4f}",
            f"{r['uniform_top1_pct']:.4f}", f"{r['tilt_top1_pct']:.4f}",
            rf"${r['tilt_minus_uniform_pp']:+.4f}\pm {r['paired_sd_pp']:.4f}$", f"{r['positive_pairs']}/4",
        ))
    return tex_table(
        r"\textbf{Matched parameters.} Top-1 after CMA on the official CIFAR tests and Tiny ImageNet "
        r"validation set. Budgets are nominal reductions in total parameters. $U$ is \Uniform{} and $T$ is "
        r"\TiltName{} (\Tilt{}); the $T-U$ column gives the paired mean and sample SD across checkpoints, in percentage "
        r"points (pp). The last column counts strictly positive differences.",
        "tilt-uniform", "llrrrrrr",
        [tex_row("", "", "", r"\multicolumn{3}{c}{Top-1 (\%)}", "", ""), r"\cmidrule(lr){4-6}",
         tex_row("Architecture", "Dataset", "$P$ cut", "Full", "$U$", "$T$", "$T-U$ (pp)", "Positive")], body,
    )


def tex_cost(rows):
    body = []
    for new, first, r in grouped_rows(rows):
        if new and not first:
            body.append(r"\addlinespace[2pt]")
        body.append(tex_row(
            r["architecture"] if new else "", r["dataset"] if new else "",
            rf"{r['anchor_nominal_parameter_reduction_pct']}\%", f"{r['uniform_top1_pct']:.3f}",
            f"{r['tilt_top1_pct']:.3f}", f"{r['wide_uniform_top1_pct']:.3f}",
            f"${r['wide_uniform_minus_tilt_pp']:+.3f}$", f"{r['tilt_over_uniform_mac']:.3f}",
            f"{r['wide_uniform_over_tilt_parameters']:.3f}", f"{r['wide_uniform_over_tilt_mac']:.3f}",
        ))
    return tex_table(
        r"\textbf{Cost and accuracy on development data.} Top-1 uses the same development "
        r"split and CMA protocol for uniform folding ($U$), \TiltName{} ($T$), and the wider uniform model $U'$. The anchor is the nominal "
        r"parameter reduction of $U$ and $T$, not of $U'$. The $U'-T$ column is the mean paired difference; "
        r"positive values favor $U'$. Section~\ref{sec:allocation-cost} summarizes forward-pass timing ratios.",
        "cost-controls", "llrrrrrrrr",
        [tex_row("", "", "", r"\multicolumn{3}{c}{Top-1 (\%)}", "", r"\multicolumn{3}{c}{Cost ratios}"),
         r"\cmidrule(lr){4-6}\cmidrule(lr){8-10}",
         tex_row("Architecture", "Dataset", "Anchor", "$U$", "$T$", "$U'$", "$U'-T$ (pp)",
                 r"\shortstack{MACs\\$T/U$}", r"\shortstack{Params.\\$U'/T$}", r"\shortstack{MACs\\$U'/T$}")],
        body, spacing=3.5,
    )


def tex_dual(rows):
    body = [tex_row(r["dataset"], f"{r['tilt_top1_pct']:.4f}", f"{r['mean_control_top1_pct']:.5f}",
                    rf"${r['tilt_minus_mean_control_pp']:+.5f}\pm {r['paired_sd_pp']:.4f}$",
                    f"{r['positive_pairs']}/4", str(r["best_control_id"]),
                    f"${r['tilt_minus_best_control_pp']:+.4f}$") for r in rows]
    return tex_table(
        r"\textbf{Both costs matched to \Tilt{}.} Four prespecified ResNet18 controls at the nominal "
        r"50\% parameter cut, each using 99.9--100\% of \Tilt{}'s parameters and MACs. $\bar C$ is the mean of "
        r"the four controls within each checkpoint. The positive count refers to $T-\bar C$. "
        r"The best control is identified descriptively by mean development top-1.",
        "dual-tilt", "lrrrrrr",
        [tex_row("", r"\multicolumn{2}{c}{Top-1 (\%)}", "", "", "", ""), r"\cmidrule(lr){2-3}",
         tex_row("Dataset", "$T$", r"$\bar C$", r"$T-\bar C$ (pp)", "Positive", "Best control", "$T-$best (pp)")],
        body,
    )


def tex_b10(rows):
    body = []
    for i, r in enumerate(rows):
        if i % 3 == 0:
            if i:
                body.append(r"\midrule")
            count = f"{r['images']:,}"
            sample = "Tiny ImageNet development split" if r["sample"] == "Tiny ImageNet dev" else r["sample"]
            body.append(tex_row(rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{sample} ({count} images)}}}}"))
            body.append(r"\addlinespace[2pt]")
        meanloss, sd = r["paired_loss_to_full_pp"], r["paired_loss_sd_pp"]
        body.append(tex_row(r["model"], f"{r['parameters']:,}", f"{r['mac_per_image']:,}",
                            rf"${r['top1_pct']:.4f}\pm {r['top1_sd_pp']:.4f}$",
                            rf"${meanloss:.4f}\pm {sd:.4f}$" if r["model_id"] != "teacher" else "0"))
    return tex_table(
        r"\textbf{Nominal 10\% parameter cut under both uniform-folding caps on earlier Tiny models.} "
        r"Cost counts are identical "
        r"across samples. For each checkpoint, top-1 loss is the full model's top-1 minus that of the "
        r"specified model on the same images. Values are mean $\pm$ sample SD across four checkpoints "
        r"after CMA. The ImageNetV2 sample is MatchedFrequency restricted to the 200 Tiny ImageNet classes.",
        "b10", "lrrrr", [tex_row("Model", "Parameters", "MACs/image", r"Top-1 (\%)", "Paired loss (pp)")], body,
        spacing=8,
    )


def tex_b10_transfer(rows):
    body = []
    for i, r in enumerate(rows):
        if i % 4 == 0:
            if i:
                body.append(r"\midrule")
            body.append(tex_row(rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{r['dataset']} ({r['sample']})}}}}"))
            body.append(r"\addlinespace[2pt]")
        top1 = f"{r['top1_pct']:.4f}"
        if r["model_id"] == "b10_L06":
            top1 = rf"\textbf{{{top1}}}"
        model = "Projected TILT" if r["model_id"] == "b10_Tproj" else r["model"]
        body.append(tex_row(model, top1, f"{r['loss_to_full_pp']:.4f}",
                            f"{r['parameters']:,}", f"{r['mac_per_image']:,}"))
    return tex_table(
        r"\textbf{Nominal 10\% parameter cut under both UNIFORM budgets on newly trained ResNet18 models.} "
        r"Each pair starts from the same checkpoint and uses the same BN recalibration, including "
        r"the full-model reference. Top-1 and loss are means across four checkpoints on 10,000 images "
        r"per dataset. Projected TILT adjusts the linear profile to both budgets. "
        r"The official test and validation splits had previously been evaluated in this project.",
        "b10-transfer", "lrrrr",
        [tex_row("Model", r"Top-1 (\%)", "Loss to full (pp)", "Parameters", "MACs/image")], body, spacing=7,
    )


def tex_tilt_new(rows):
    body = []
    for i, r in enumerate(rows):
        if i % 3 == 0:
            if i:
                body.append(r"\midrule")
            cohort = f"{r['dataset']} ({r['sample']}; full top-1 {r['full_top1_pct']:.4f}\\%)"
            body.append(tex_row(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{cohort}}}}}"))
            body.append(r"\addlinespace[2pt]")
        body.append(tex_row(rf"{r['nominal_parameter_reduction_pct']}\%", f"{r['uniform_top1_pct']:.4f}",
                            f"{r['tilt_top1_pct']:.4f}", rf"${r['tilt_minus_uniform_pp']:+.4f}$",
                            f"{r['tilt_loss_to_full_pp']:.4f}", f"{r['tilt_over_uniform_mac']:.3f}"))
    return tex_table(
        r"\textbf{Width allocation at matched parameter counts on newly trained ResNet18 checkpoints.} "
        r"$U$ is uniform folding and $T$ is Linear Depth Tilt. $T$ uses 99.9--100\% of $U$'s parameters. "
        r"Top-1 and losses are means across four paired checkpoints after the same BN procedure. "
        r"All six $T-U$ contrasts are positive on all four checkpoints. "
        r"$M(T)/M(U)$ is the ratio of MACs per image, not a runtime ratio. Both benchmarks had "
        r"already been used earlier in the project.",
        "tilt-new", "lrrrrr",
        [tex_row("", r"\multicolumn{2}{c}{Top-1 (\%)}", "", "", ""), r"\cmidrule(lr){2-3}",
         tex_row("Nominal $P$ cut", "$U$", "$T$", "$T-U$ (pp)", "$T$ loss to full (pp)", "$M(T)/M(U)$")],
        body, spacing=6,
    )


def tex_tilt_mild(rows):
    body = []
    for i, r in enumerate(rows):
        if i % 2 == 0:
            if i:
                body.append(r"\midrule")
            cohort = f"{r['dataset']} (full top-1 {r['full_top1_pct']:.4f}\\%)"
            body.append(tex_row(rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{cohort}}}}}"))
            body.append(r"\addlinespace[2pt]")
        body.append(tex_row(rf"{r['nominal_parameter_reduction_pct']}\%", f"{r['uniform_top1_pct']:.4f}",
                            f"{r['tilt_top1_pct']:.4f}", rf"${r['tilt_minus_uniform_pp']:+.4f}$",
                            f"{r['tilt_loss_to_full_pp']:.4f}", f"{r['tilt_parameter_reduction_pct']:.2f}",
                            f"{r['tilt_mac_reduction_pct']:.2f}", f"{r['tilt_over_uniform_mac']:.3f}"))
    return tex_table(
        r"\textbf{Mild cuts at matched parameter counts (development diagnostic).} "
        r"$T$ uses 99.9--100\% of $U$'s parameters; MACs are reported rather than capped. "
        r"Mean top-1 after the same BN procedure within each dataset; all differences favor $T$ on all four "
        r"checkpoints. Parameter and MAC reductions are calculated relative to the full model.",
        "tilt-mild-dev", "lrrrrrrr",
        [tex_row("", r"\multicolumn{2}{c}{Top-1 (\%)}", "", "", r"\multicolumn{2}{c}{Reductions for $T$ (\%)}", ""),
         r"\cmidrule(lr){2-3}\cmidrule(lr){6-7}",
         tex_row("Nominal $P$ cut", "$U$", "$T$", "$T-U$ (pp)", "$T$ loss (pp)", "$P$", "MACs", "$M(T)/M(U)$")],
        body, spacing=5,
    )


def main():
    if not __debug__:
        raise RuntimeError("Uruchom bez -O: weryfikacja wejść używa assert.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Sprawdź wygenerowane pliki bez zapisu.")
    args = parser.parse_args()
    e035, plans, score = pair_rows_e035()
    cost = cost_rows(plans, score)
    dual = dual_rows()
    b10, per_seed, contrasts = b10_rows()
    transfer, transfer_seeds, transfer_contrasts = b10_transfer_rows()
    crop = b10_crop_rows(transfer_seeds)
    tilt_new = tilt_new_rows()
    tilt_mild = tilt_mild_rows()
    report(e035, cost, dual, b10, contrasts)
    for r in transfer_contrasts:
        if r["sample"] != "development":
            print(f"E049 {r['dataset']} {r['comparison']}: {r['mean_pp']:+.4f} pp, {r['positive_pairs']}/4.")
    print(f"E050 Tiny dev: L06 shift {crop[-1]['change_pp']:+.4f} pp at the same costs, 4/4.")
    print("E049 at matched P: 6/6 positive contrasts on new checkpoints, 4/4 each.")
    print("E051 dev: b10 passes the practical criterion on CIFAR-100 only; b5 reported separately.")
    out = PAPER / "tables"
    if not args.check:
        out.mkdir(exist_ok=True)
    render = {"tilt_uniform": (e035, tex_e035), "cost_controls": (cost, tex_cost),
              "dual_cost_controls": (dual, tex_dual), "b10": (b10, tex_b10),
              "b10_per_seed": (per_seed, None), "b10_contrasts": (contrasts, None),
              "b10_transfer": (transfer, tex_b10_transfer), "b10_transfer_per_seed": (transfer_seeds, None),
              "b10_transfer_contrasts": (transfer_contrasts, None), "b10_tiny_crop": (crop, None),
              "tilt_new": (tilt_new, tex_tilt_new), "tilt_mild_dev": (tilt_mild, tex_tilt_mild)}
    for stem, (data, tex) in render.items():
        outputs = {"csv": csv_text(data)}
        if tex is not None:
            outputs["tex"] = tex(data)
        for ext, text in outputs.items():
            path = out / f"{stem}.{ext}"
            if args.check:
                assert path.read_text(encoding="utf-8") == text, path
            else:
                path.write_text(text, encoding="utf-8")
    print(f"PASS: tables and CSVs {'checked' if args.check else 'written'} in {out.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
