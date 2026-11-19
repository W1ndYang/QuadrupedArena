"""Estimate independent holdout size from a genuine paired pilot archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tarfile

import numpy as np
from scipy.stats import binom, binomtest

from llm_compare_ubuntu import verify_comparison
from llm_study_data import emit
from llm_study_online import excluded_seed_streams, primary_comparisons
from llm_study_protocol import digest


def exact_power(n, discordance, difference, alpha):
    """Unconditional probability of rejection by the paired exact test."""
    if not 0 < difference <= discordance <= 1:
        raise ValueError('Expected 0 < minimum difference <= discordance <= 1')
    counts = np.arange(n+1)
    lower = binom.ppf(alpha/2, counts, .5).astype(int)
    lower -= binom.cdf(lower, counts, .5) > alpha/2
    alternative = (discordance+difference)/(2*discordance)
    rejection = (binom.cdf(lower, counts, alternative) +
                 binom.sf(counts-lower-1, counts, alternative))
    return float(np.dot(binom.pmf(counts, n, discordance), rejection))


def sample_size(discordance, effect, alpha, power, maximum):
    for n in range(4, maximum+1):
        achieved = exact_power(n, discordance, effect, alpha)
        if achieved >= power:
            return dict(n=n, achieved_power=achieved)
    return dict(n=None, power_at_maximum=exact_power(maximum, discordance, effect, alpha))


def from_pilot(path, effect, alpha, power, maximum):
    verify_comparison(path)
    with tarfile.open(path) as archive:
        pilot = json.load(archive.extractfile('llm_comparison/comparison.json'))
    if pilot['phase'] != 'pilot' or pilot['fixture_only'] or pilot['status'] != 'completed':
        raise ValueError('Power calculation requires a completed, live paired pilot')
    tests = primary_comparisons(pilot['methods'])
    expected = len(pilot['seed_pairs'])
    if len(pilot['rows']) != expected*len(pilot['methods']):
        raise ValueError('Pilot matrix does not contain every planned episode')
    rows = []
    for name in tests:
        pair = pilot['aggregate']['paired'][name]
        changed = pair['reference_only']+pair['comparison_only']
        n = pair['n']
        if n != expected:
            raise ValueError(f'Incomplete pairing for {name}')
        rate = changed/n
        interval = binomtest(changed, n).proportion_ci(confidence_level=.95)
        # Pilot discordance can be estimated imprecisely. Include its upper
        # limit as a separate, less favorable planning scenario.
        scenarios = sorted({max(effect, rate), max(effect, float(interval.high))})
        plans = [dict(discordance=q, **sample_size(q, effect, alpha/len(tests), power, maximum))
                 for q in scenarios]
        rows.append(dict(comparison=name, pilot_pairs=n, discordant_pairs=changed,
                         pilot_discordance_95=[interval.low, interval.high], scenarios=plans))
    finite = [case['n'] for row in rows for case in row['scenarios'] if case['n'] is not None]
    all_estimated = len(finite) == sum(len(row['scenarios']) for row in rows)
    return dict(pilot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                pilot_seed_pairs=pilot['seed_pairs'], direct_model=pilot['direct_model'],
                methods=pilot['methods'], decision_config_sha256=pilot['decision_config_sha256'],
                decision_runtime_version=pilot['decision_runtime_version'],
                minimum_detectable_absolute_difference=effect, family_alpha=alpha,
                per_test_alpha=alpha/len(tests), desired_power=power,
                maximum_considered=maximum, comparisons=rows,
                suggested_common_seed_count=max(finite) if all_estimated else None,
                interpretation='Paired exact test planning against the declared effect, '
                               'using pilot discordance and its upper interval. '
                               'Generate new seeds after freezing the planned size.')


def freeze_holdout(power_report, output):
    n = power_report['suggested_common_seed_count']
    if n is None:
        raise ValueError('Power target is unresolved. No holdout seed plan was created')
    first, second = excluded_seed_streams('holdout')
    for a,b in power_report['pilot_seed_pairs']:
        first.add(a)
        second.add(b)
    key = digest({'pilot': power_report['pilot_sha256'],
                  'power_design': digest(power_report), 'phase': 'holdout'})
    generator = random.Random(int(key,16))

    def draw(used):
        while True:
            seed = generator.randrange(2**32)
            if seed not in used:
                used.add(seed)
                return seed

    plan = dict(schema='llm-study-seeds-v1', phase='holdout',
                pilot_sha256=power_report['pilot_sha256'], power_report_sha256=digest(power_report),
                direct_model=power_report['direct_model'], methods=power_report['methods'],
                decision_config_sha256=power_report['decision_config_sha256'],
                decision_runtime_version=power_report['decision_runtime_version'],
                created_utc=datetime.now(timezone.utc).isoformat(),
                seed_pairs=[[draw(first), draw(second)] for _ in range(n)])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(plan,handle,indent=2)
        handle.write('\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-effect', type=float, default=.15)
    parser.add_argument('--power', type=float, default=.8)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--maximum', type=int, default=500)
    parser.add_argument('--new-holdout-plan', type=Path)
    args = parser.parse_args()
    if not (0 < args.minimum_effect < 1 and 0 < args.power < 1 and
            0 < args.alpha < 1 and args.maximum >= 4):
        parser.error('Invalid power planning inputs')
    report = from_pilot(args.pilot, args.minimum_effect, args.alpha, args.power, args.maximum)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emit(args.output, report)
    if args.new_holdout_plan:
        freeze_holdout(report,args.new_holdout_plan)
    print(json.dumps(report, indent=2))
