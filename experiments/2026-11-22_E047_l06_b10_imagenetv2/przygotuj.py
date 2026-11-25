"""2026-11-22_E047_l06_b10_imagenetv2: źródło, mapowanie i audyt rozłączności CPU.
Czyta wydanie ImageNetV2 i cache Tiny; utrwala listę i cache nowych wejść bez
predykcji. Reguły w OPIS.md, zapis pod runs/E047/cpu_<UTC>_<id>.
"""

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs/E047'
E046 = HERE.parent / '2026-11-22_E046_l06_b10_tiny'
sys.path.insert(0, str(E046))
import ocena_dev as e46

cpu = e46.cpu
read, write, digest, rel = e46.read, e46.write, e46.digest, e46.rel
HF_REV = 'd626240be2538720e83103a0e1178d24aca8b12c'
URL = f'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/{HF_REV}/imagenetv2-matched-frequency.tar.gz'
ARCHIVE_BYTES = 1264079360
ARCHIVE_SHA = 'f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c'
REV = '0b0c3fe89058e4c7ee2348938b3f4f706a931bf3'
META = {
    'metadata/class_info.json': '01968f45aa507b6ccf4d9a1f5f71f1dfc16819f32d6d84ee95861edba98d043b',
    'datasets/imagenetv2-b-33.json': 'ca1ebce4c2f450835a8640ed447304d3d4318647f3930d7ba752b12ee692728d',
}
TRAIN, VAL = e46.CACHE, ROOT / 'data/tiny-imagenet/cache_val.npz'
VAL_SHA = 'fd0c5fc8e45cff072f45f878b60a33d8f0e0af6340fc11d8f937e1c6c75bf07b'
WNIDS = ROOT / 'data/tiny-imagenet/tiny-imagenet-200/wnids.txt'
MAX_HAMMING, MIN_TOTAL, MIN_CLASS = 4, 1800, 8
DATA_FILE = 'heldout.npz'


def new_run(prefix=''):
    run = RUNS / f'{prefix}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    (run / 'source').mkdir(parents=True)
    return run


def snapshot(run, names):
    for name in names:
        shutil.copyfile(HERE / name, run / 'source' / name)
    return {name: digest(run / 'source' / name) for name in names}


def metadata(run):
    parsed = []
    for name, expected in META.items():
        url = f'https://raw.githubusercontent.com/modestyachts/ImageNetV2/{REV}/data/{name}'
        with urllib.request.urlopen(url, timeout=45) as response:
            blob = response.read(2_000_001)
        assert hashlib.sha256(blob).hexdigest() == expected, url
        (run / Path(name).name).write_bytes(blob)
        parsed.append(json.loads(blob))
    classes, source = parsed
    assert source['is_valid'] and source['sampling_function'] == 'sample_wnid_histogram'
    assert source['target_size'] == 10000 and source['output_filename'] == 'imagenetv2-b-33.json'
    cid_wnid = {int(v['cid']): v['wnid'] for v in classes}
    image_wnid = dict(source['image_filenames'])
    assert len(classes) == len(cid_wnid) == len(set(cid_wnid.values())) == 1000
    assert set(cid_wnid) == set(range(1000))
    assert len(image_wnid) == len(source['image_filenames']) == 10000
    assert Counter(image_wnid.values()) == {wnid: 10 for wnid in cid_wnid.values()}
    return cid_wnid, image_wnid


def dataset(run, classes, np, image, ops):
    cid_wnid, image_wnid = metadata(run)
    targets = {wnid: i for i, wnid in enumerate(classes)}
    assert targets.keys() <= set(cid_wnid.values())
    seen, selected = set(), []
    sha, size = hashlib.sha256(), 0
    with tempfile.TemporaryFile(dir='/tmp/opencode') as file:
        with urllib.request.urlopen(URL, timeout=60) as response:
            assert int(response.headers['Content-Length']) == ARCHIVE_BYTES
            for chunk in iter(lambda: response.read(8 << 20), b''):
                size += len(chunk)
                assert size <= ARCHIVE_BYTES
                sha.update(chunk)
                file.write(chunk)
        assert size == ARCHIVE_BYTES and sha.hexdigest() == ARCHIVE_SHA
        file.seek(0)
        print('Archiwum zgodne z przypiętym SHA-256; sprawdzam etykiety.', flush=True)
        with tarfile.open(fileobj=file, mode='r|*') as tar:
            for member in tar:
                if member.isdir():
                    continue
                path = PurePosixPath(member.name)
                assert member.isfile() and len(path.parts) == 3, member.name
                assert path.parts[0] == 'imagenetv2-matched-frequency-format-val'
                assert path.suffix.lower() in ('.jpeg', '.jpg')
                wnid, ident = cid_wnid[int(path.parent.name)], path.stem
                assert ident not in seen and image_wnid[ident] == wnid
                seen.add(ident)
                if wnid not in targets:
                    continue
                assert 0 < member.size < 40 << 20
                with tar.extractfile(member) as stream:
                    blob = stream.read()
                with image.open(io.BytesIO(blob)) as src:
                    rgb = ops.exif_transpose(src).convert('RGB')
                shape = rgb.size
                resized = np.array(rgb.resize((64, 64), image.Resampling.BILINEAR))
                w, h = shape
                side = min(w, h)
                left, top = (w - side) // 2, (h - side) // 2
                crop = rgb.crop((left, top, left+side, top+side))
                center = np.array(crop.resize((64, 64), image.Resampling.BILINEAR))
                selected.append((targets[wnid], ident, resized, center, hashlib.sha256(blob).hexdigest()))
    assert seen == image_wnid.keys() and Counter(row[0] for row in selected) == {i: 10 for i in range(200)}
    print('2 000 obrazów, zgodne etykiety dwóch źródeł.', flush=True)
    return sorted(selected, key=lambda row: (row[0], row[1]))


def audit_overlap(selected, expected, np, image, dctn):
    def phash(images):
        blocks = np.stack([np.asarray(image.fromarray(im).convert('L').resize((32, 32), image.Resampling.LANCZOS))
                           for im in images]).astype(np.float64)
        freq = dctn(blocks, axes=(1, 2), norm='ortho')[:, :8, :8].reshape(-1, 64)[:, 1:]
        bits = (freq > np.median(freq, axis=1, keepdims=True)).astype(np.uint64)
        return np.sum(bits << np.arange(63, dtype=np.uint64), axis=1, dtype=np.uint64)

    pixels = np.stack([row[2] for row in selected])
    center = np.stack([row[3] for row in selected])
    variants = [pixels, pixels[:, :, ::-1, :], center, center[:, :, ::-1, :]]
    test_hash = np.stack([phash(v) for v in variants])
    exact = np.asarray([[hashlib.sha256(im.tobytes()).hexdigest() for im in v] for v in variants])
    known_refs, rgb_hits, known_hash = {}, set(), []
    digests = set(exact.ravel())
    for p, count in ((TRAIN, 100000), (VAL, 10000)):
        assert digest(p) == expected[rel(p)]
        with np.load(p, allow_pickle=False) as cache:
            imgs = cache['images']
        assert imgs.shape == (count, 64, 64, 3) and imgs.dtype == np.uint8
        for offset in range(0, count, 512):
            rows = imgs[offset:offset+512]
            known_hash.append(phash(rows))
            for im in rows:
                h = hashlib.sha256(im.tobytes()).hexdigest()
                if h in digests:
                    rgb_hits.add(h)
        known_refs[rel(p)] = dict(count=count, sha256=expected[rel(p)])
        del imgs
    known = np.concatenate(known_hash)
    assert len(known) == 110000
    overlap, pairs, entries = set(), [], []
    for i, (label, ident, _, _, jpeg_sha) in enumerate(selected):
        distances = np.bitwise_count(test_hash[:, i, None] ^ known)
        min_dist = int(distances.min())
        exact_match = bool(rgb_hits.intersection(exact[:, i]))
        if min_dist <= MAX_HAMMING or exact_match:
            overlap.add(i)
        pair_distances = np.bitwise_count(test_hash[:, i, None] ^ test_hash[0]).min(axis=0)
        peers = (pair_distances <= MAX_HAMMING) | np.any(exact[:, i, None] == exact[0], axis=0)
        peers[i] = False
        for j in np.flatnonzero(peers):
            pairs.append((i, int(j)))
        entries.append(dict(id=ident, label=label, jpeg_sha256=jpeg_sha, rgb_sha256=exact[0, i].item(),
                            min_tiny_hamming=min_dist, tiny_exact_match=exact_match))
    within = {i for pair in pairs for i in pair}
    keep = np.asarray([i not in overlap | within for i in range(len(selected))])
    for i, item in enumerate(entries):
        item.update(retained=bool(keep[i]), similar_within=bool(i in within))
    audit = dict(known_sources=known_refs, max_hamming=MAX_HAMMING, before=len(selected), after=int(keep.sum()),
                 overlap_tiny=len(overlap), overlap_tiny_exact=sum(row['tiny_exact_match'] for row in entries),
                 flagged_within=len(within), pairs_within=sorted({tuple(sorted(p)) for p in pairs}))
    return pixels[keep], np.asarray([row[0] for row in selected], dtype=np.int64)[keep], entries, audit


def main():
    assert len(sys.argv) == 1 and not sys.flags.optimize and sys.dont_write_bytecode
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    run, started = new_run('cpu_'), time.monotonic()
    m = dict(status='running', phase='source_and_overlap_cpu', run=rel(run), started_utc=e46.utc(), pid=os.getpid(),
             command=[sys.executable, '-B', *sys.argv], image_evaluation=False,
             source_sha256=snapshot(run, ('przygotuj.py', 'OPIS.md')), python=sys.version,
             archive=dict(url=URL, bytes=ARCHIVE_BYTES, sha256=ARCHIVE_SHA), official_metadata_sha256=META)
    write(run / 'manifest.json', m)
    print('CPU RUN', rel(run), flush=True)
    opened, denied = e46.guard_files({TRAIN, VAL, WNIDS})
    try:
        import numpy as np
        import scipy
        import PIL
        from PIL import Image, ImageOps
        from scipy.fft import dctn
        m.update(numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__)
        hashes = {}
        f45 = cpu.read(cpu.R45 / 'frozen.json', cpu.F45_SHA, hashes)
        expect = f45['input_sha256']
        cpu.verify(WNIDS, expect[rel(WNIDS)], hashes)
        e19 = cpu.R19 / 'data_manifest.json'
        classes = sorted(WNIDS.read_text().splitlines())
        assert len(set(classes)) == len(classes) == 200
        assert classes == cpu.read(e19, expect[rel(e19)], hashes)['classes']
        f46 = cpu.read(e46.CPU_RUN / 'frozen.json', e46.FROZEN_SHA, hashes)
        assert f46['calibration']['sha256'] == f45['calibration']['sha256']
        old = {rel(TRAIN): f46['calibration']['sha256'], rel(VAL): VAL_SHA}
        selected = dataset(run, classes, np, Image, ImageOps)
        pixels, labels, entries, audit = audit_overlap(selected, old, np, Image, dctn)
        counts = np.bincount(labels, minlength=200)
        passed = len(labels) >= MIN_TOTAL and bool(np.all(counts >= MIN_CLASS))
        write(run / 'cohort.json', dict(classes=classes, entries=entries, counts=counts.tolist()))
        write(run / 'audit.json', dict(status='PASS' if passed else 'STOP', counts=counts.tolist(), **audit))
        assert opened == {TRAIN, VAL, WNIDS} and not denied
        if not passed:
            m['status'] = 'stopped_coverage'
            print('STOP: pokrycie próby poniżej zamrożonej bramki', flush=True)
            return
        np.savez_compressed(run / DATA_FILE, images=pixels, labels=labels)
        files = (DATA_FILE, 'cohort.json', 'audit.json', *(Path(name).name for name in META))
        m.update(status='passed', samples=len(labels), output_sha256={n: digest(run / n) for n in files},
                 input_sha256=hashes, model_plan_sha256=e46.FROZEN_SHA)
        print('PASS', len(labels), 'obrazów; wyłączenia:', audit['before']-audit['after'],
              'Tiny:', audit['overlap_tiny'], 'wewnątrz V2:', audit['flagged_within'], flush=True)
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        m.update(finished_utc=e46.utc(), seconds=time.monotonic()-started,
                 opened=sorted(map(rel, opened)), denied=denied)
        write(run / 'manifest.json', m)


if __name__ == '__main__':
    main()
