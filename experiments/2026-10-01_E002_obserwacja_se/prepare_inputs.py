"""2026-10-01_E002_obserwacja_se: ustala listy obrazów przed pomiarem (CPU).
Wejścia: config.draft.json, zweryfikowane manifesty E001 i train CIFAR-100.
Wyjścia: nowe SAMPLES.json i INPUTS.json; odmawia nadpisania istniejących.
Protokół: PROTOKOL_v1.md; szkic: OPIS.md. Nie uruchamia inferencji.
"""
from common import (HERE, ROOT, digest, inspect_inputs, read_json, require,
                    sample_manifest, training_container, verify_samples, write_json)


def main():
    cfg = read_json(HERE / 'config.draft.json')
    require(not (HERE / 'SAMPLES.json').exists() and not (HERE / 'INPUTS.json').exists(),
            'Input manifests already exist; refusing to overwrite')
    integrity = inspect_inputs(cfg)
    _, labels = training_container()
    samples = sample_manifest(cfg, labels)
    verify_samples(cfg, samples, labels)
    write_json(HERE / 'SAMPLES.json', samples)
    integrity['samples_sha256'] = digest(HERE / 'SAMPLES.json')
    write_json(HERE / 'INPUTS.json', integrity)
    print(f"Samples: A=500 B=500; SHA256={integrity['samples_sha256']}")
    print('Input files verified; no inference, no GPU, no test/reserve/dev evaluation.')


if __name__ == '__main__':
    main()
