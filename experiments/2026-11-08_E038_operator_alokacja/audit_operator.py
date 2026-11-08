"""2026-11-08_E038_operator_alokacja: dodatkowy audyt MAG2 po runie; OPIS.md.
Wejście: run i oryginalny upstream MAG; wyjście: porównanie tensorów120 modeli.
Niezależna ścieżka przebudowy upstream, bez GPU, danych i nowych ocen jakości.
"""
import copy
import sys
from pathlib import Path

import runtime as rt
from pruning import prune


def audit(run):
    assert not rt.torch.cuda.is_available()
    rt.torch.set_num_threads(2)
    _, frozen = rt.verify_run(run)
    from compression.mag_prune import ResNet18_MagnitudePruning, PreActResNet18_MagnitudePruning
    rows = []
    for pair in rt.PAIRS:
        preact = pair.startswith('preact')
        axes = (rt.e35.e34.get_axis_to_perm_PreActResNet18(override=False) if preact
                else rt.e35.e29.get_axis_to_perm_ResNet18(override=False))
        group_names = list(rt.e35.e29.axes2perm_to_perm2axes(axes))
        cls = PreActResNet18_MagnitudePruning if preact else ResNet18_MagnitudePruning
        for seed in rt.SEEDS:
            base = rt.e35.model_source(pair, seed)
            for tag in rt.TAGS:
                summary = rt.read(run / f'seed_{seed}/{pair}/{tag}/summary.json')
                for profile in ('U', 'T'):
                    widths = frozen['plans'][pair][tag][profile]
                    assignment = dict(zip(group_names, widths))
                    # Niezależna lista wywołań oryginalnego apply(), nie diagnostyka adaptera.
                    queue = ['conv1']
                    for stage in range(1, 5):
                        for block in range(2):
                            path = f'layer{stage}.{block}'
                            queue.append(path + '.conv1')
                            module = base.get_submodule(path)
                            shortcut = getattr(module, 'shortcut' if preact else 'downsample', None)
                            if isinstance(shortcut, rt.torch.nn.Sequential) and len(shortcut) > 0:
                                queue.append(path + '.conv2')
                    pending = list(queue)
                    operator = cls(copy.deepcopy(base), compression_ratio=.5, p=2)

                    def select(scores, ignored_k):
                        anchor = pending.pop(0)
                        weight = base.get_submodule(anchor).weight.detach()
                        expected = rt.torch.norm(weight.reshape(weight.shape[0], -1), p=2, dim=1)
                        assert rt.torch.equal(scores, expected), (pair, anchor)
                        k = assignment[axes[anchor + '.weight'][0]]
                        return rt.torch.argsort(scores, descending=True, stable=True)[:k]

                    operator._get_keep_indices = select
                    reference = operator.apply().eval().requires_grad_(False)
                    assert not pending
                    actual, _ = prune(base, pair, widths, rt)
                    expected_state, state = reference.state_dict(), actual.state_dict()
                    assert expected_state.keys() == state.keys()
                    for name in state:
                        assert rt.torch.equal(state[name], expected_state[name]), (pair, seed, tag, profile, name)
                        assert rt.torch.isfinite(state[name]).all()
                    assert rt.e35.bn.weights_hash(actual) == summary['evidence'][profile]['PRE']['weights_sha256']
                    rows.append(dict(pair=pair, seed=seed, tag=tag, profile=profile, tensors=len(state), exact=True))
            print('OPERATOR EXACT', pair, seed, flush=True)
    assert len(rows) == 120 and not rt.DENIED
    rt.write(run / 'analysis/operator_verification.json', dict(status='passed', utc=rt.utc(), rows=rows,
        note='Audyt techniczny po runie: oryginalne apply MAG, adapter wyłącznie liczby k i remisów.',
        code_sha256=rt.digest(Path(__file__)), data_open_attempts=rt.DENIED))
    print('OPERATOR AUDIT PASS', len(rows), flush=True)


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
