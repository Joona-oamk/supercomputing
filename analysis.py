import argparse
import importlib
import json
import os
import signal
import statistics
import sys
import time
from collections import Counter
from math import ceil
from multiprocessing import cpu_count, get_context
from pathlib import Path

for thread_var in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(thread_var, '1')

import numpy as np

try:
    MPI = importlib.import_module('mpi4py.MPI')
except ImportError:
    MPI = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fixed-workload multiprocessing analysis for .npy image arrays.'
    )
    parser.add_argument('input_path', nargs='?', default='super_data')
    parser.add_argument('--workers', type=int, default=default_worker_count())
    parser.add_argument(
        '--repeats',
        type=int,
        default=7,
        help='Number of multi-scale analysis passes per file.',
    )
    parser.add_argument(
        '--report-json',
        type=Path,
        default=None,
        help='Optional path for a machine-readable report.',
    )
    parser.add_argument(
        '--progress-seconds',
        type=float,
        default=5.0,
        help='How often to print progress updates during calibration and analysis.',
    )
    parser.add_argument(
        '--chunk-files',
        type=int,
        default=None,
        help='How many files to hand to a worker at once. Default is chosen automatically.',
    )
    return parser.parse_args()


def default_worker_count():
    detected = detected_cpu_budget()
    if detected <= 2:
        return 1
    return detected - 1


def detected_cpu_budget():
    for env_name in ('SLURM_CPUS_PER_TASK', 'PBS_NCPUS'):
        value = os.environ.get(env_name)
        if value and value.isdigit():
            return max(1, int(value))
    return cpu_count() or 1


def get_mpi_context():
    if MPI is None:
        return {
            'enabled': False,
            'comm': None,
            'rank': 0,
            'size': 1,
            'local_rank': 0,
        }

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    local_rank = int(
        os.environ.get('SLURM_LOCALID', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    )
    return {
        'enabled': size > 1,
        'comm': comm,
        'rank': rank,
        'size': size,
        'local_rank': local_rank,
    }


def log(message, prefix=''):
    print(f'{prefix}{message}', flush=True)


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def split_files_for_rank(files, rank, size):
    return files[rank::size]


def compute_kernel(array, repeats):
    work = array.astype(np.float64, copy=False)
    edge_accumulator = 0.0
    spatial_corr_accumulator = 0.0
    high_freq_accumulator = 0.0
    hotspot_accumulator = 0.0
    entropy_accumulator = 0.0

    height, width = work.shape
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt((freq_y * freq_y) + (freq_x * freq_x))
    high_freq_mask = radius >= 0.22

    for _ in range(repeats):
        north = np.roll(work, 1, axis=0)
        south = np.roll(work, -1, axis=0)
        west = np.roll(work, 1, axis=1)
        east = np.roll(work, -1, axis=1)

        local_mean = (work + north + south + west + east) / 5.0
        gradient_x = east - west
        gradient_y = south - north
        residual = work - local_mean

        edge_energy = float(np.mean((gradient_x * gradient_x) + (gradient_y * gradient_y)))
        centered = work - float(np.mean(work))
        variance = float(np.mean(centered * centered)) + 1e-12
        corr_x = float(np.mean(centered * np.roll(centered, 1, axis=1)) / variance)
        corr_y = float(np.mean(centered * np.roll(centered, 1, axis=0)) / variance)
        spatial_corr = 0.5 * (corr_x + corr_y)

        spectrum = np.abs(np.fft.rfft2(residual)) ** 2
        spectrum_total = float(np.sum(spectrum)) + 1e-12
        high_frequency_ratio = float(np.sum(spectrum[high_freq_mask]) / spectrum_total)

        residual_std = float(np.std(residual)) + 1e-12
        hotspot_fraction = float(np.mean(np.abs(residual) > (2.5 * residual_std)))

        clipped = np.clip(work, 0.0, 255.0)
        histogram, _ = np.histogram(clipped, bins=32, range=(0.0, 255.0))
        probabilities = histogram.astype(np.float64) / max(int(np.sum(histogram)), 1)
        nonzero_probabilities = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(nonzero_probabilities * np.log2(nonzero_probabilities)))

        edge_accumulator += edge_energy
        spatial_corr_accumulator += spatial_corr
        high_freq_accumulator += high_frequency_ratio
        hotspot_accumulator += hotspot_fraction
        entropy_accumulator += entropy

        work = (0.65 * local_mean) + (0.35 * np.sqrt(np.abs(residual) + 1.0))

    edge_mean = edge_accumulator / repeats
    spatial_corr_mean = spatial_corr_accumulator / repeats
    high_freq_ratio_mean = high_freq_accumulator / repeats
    hotspot_fraction_mean = hotspot_accumulator / repeats
    entropy_mean = entropy_accumulator / repeats
    texture_score = edge_mean * (1.0 + abs(spatial_corr_mean)) * (1.0 - high_freq_ratio_mean)
    anomaly_score = hotspot_fraction_mean * 1000.0

    return {
        'texture_score': texture_score,
        'anomaly_score': anomaly_score,
        'edge_mean': edge_mean,
        'spatial_corr_mean': spatial_corr_mean,
        'high_freq_ratio_mean': high_freq_ratio_mean,
        'hotspot_fraction_mean': hotspot_fraction_mean,
        'entropy_mean': entropy_mean,
    }


def worker_initializer():
    for signame in ('SIGINT', 'SIGBREAK'):
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), signal.SIG_IGN)

    prefix = os.environ.get('ANALYSIS_LOG_PREFIX', '')
    print(
        f'{prefix}[worker-start] '
        f'pid={os.getpid()} '
        f'omp={os.environ.get("OMP_NUM_THREADS", "unset")} '
        f'openblas={os.environ.get("OPENBLAS_NUM_THREADS", "unset")} '
        f'mkl={os.environ.get("MKL_NUM_THREADS", "unset")}',
        flush=True,
    )


def analyze_file(task):
    file_path_str, repeats = task
    file_path = Path(file_path_str)

    start_time = time.perf_counter()
    array = np.load(file_path)
    kernel_result = compute_kernel(array, repeats)
    elapsed = time.perf_counter() - start_time

    return {
        'file': file_path.name,
        'rank': int(os.environ.get('ANALYSIS_MPI_RANK', '0')),
        'pid': os.getpid(),
        'duration_seconds': elapsed,
        'shape': list(array.shape),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        **kernel_result,
    }


def emit_progress(phase, completed, total, started_at, results, known_pids):
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    throughput = completed / elapsed
    percent = (100.0 * completed) / total
    mean_task = statistics.fmean(result['duration_seconds'] for result in results)
    print(
        f'[{phase}] '
        f'{completed}/{total} files '
        f'({percent:5.1f}%) | '
        f'elapsed={elapsed:7.2f}s | '
        f'throughput={throughput:6.2f} files/s | '
        f'mean_file={mean_task:5.3f}s | '
        f'active_workers={len(known_pids)}',
        flush=True,
    )


def run_pool(pool, files, repeats, workers, phase, progress_seconds, chunk_size=None):
    tasks = [(str(file_path), repeats) for file_path in files]
    chunk_size = chunk_size or max(1, len(tasks) // (workers * 4))
    results = []
    known_pids = set()
    started_at = time.perf_counter()
    next_progress_at = started_at + max(0.5, progress_seconds)
    progress_step = max(1, ceil(len(tasks) / 20))

    print(
        f'[{phase}] Dispatching {len(tasks)} files with chunk_size={chunk_size}. '
        'Workers pull a new chunk after finishing the previous one.',
        flush=True,
    )

    for result in pool.imap_unordered(analyze_file, tasks, chunksize=chunk_size):
        results.append(result)

        if result['pid'] not in known_pids:
            known_pids.add(result['pid'])
            print(
                f'[{phase}] worker pid {result["pid"]} reported first result '
                f'({len(known_pids)}/{workers} workers seen)',
                flush=True,
            )

        completed = len(results)
        now = time.perf_counter()
        should_report = (
            completed == len(tasks)
            or completed % progress_step == 0
            or now >= next_progress_at
        )
        if should_report:
            emit_progress(phase, completed, len(tasks), started_at, results, known_pids)
            next_progress_at = now + max(0.5, progress_seconds)

    return results


def z_scores(values):
    mean_value = statistics.fmean(values)
    stddev = statistics.pstdev(values)
    if stddev < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean_value) / stddev for value in values]


def aggregate_results(results, total_runtime, workers, repeats):
    durations = [result['duration_seconds'] for result in results]
    texture_scores = [result['texture_score'] for result in results]
    anomaly_scores = [result['anomaly_score'] for result in results]
    spatial_corr = [result['spatial_corr_mean'] for result in results]
    high_freq_ratio = [result['high_freq_ratio_mean'] for result in results]
    hotspot_fraction = [result['hotspot_fraction_mean'] for result in results]
    entropy = [result['entropy_mean'] for result in results]

    texture_z = z_scores(texture_scores)
    anomaly_z = z_scores(anomaly_scores)
    spatial_z = z_scores(spatial_corr)
    high_freq_z = z_scores(high_freq_ratio)
    hotspot_z = z_scores(hotspot_fraction)
    entropy_z = z_scores(entropy)

    for index, result in enumerate(results):
        result['outlier_score'] = (
            abs(texture_z[index])
            + abs(anomaly_z[index])
            + abs(spatial_z[index])
            + abs(high_freq_z[index])
            + abs(hotspot_z[index])
            + abs(entropy_z[index])
        )

    strongest_texture = max(results, key=lambda item: item['texture_score'])
    strongest_outlier = max(results, key=lambda item: item['outlier_score'])
    slowest_file = max(results, key=lambda item: item['duration_seconds'])
    mean_abs_corr = statistics.fmean(abs(value) for value in spatial_corr)
    worker_counts = Counter((result.get('rank', 0), result['pid']) for result in results)
    worker_load = [
        {
            'rank': rank,
            'pid': pid,
            'files': count,
        }
        for (rank, pid), count in worker_counts.most_common()
    ]

    if mean_abs_corr < 0.01:
        dataset_note = 'Frames are mostly noise-like: neighboring pixels have almost no correlation.'
    elif mean_abs_corr < 0.05:
        dataset_note = 'Frames contain weak local structure, but noise still dominates.'
    else:
        dataset_note = 'Frames show visible spatial structure beyond pure noise.'

    return {
        'file_count': len(results),
        'workers': workers,
        'repeats': repeats,
        'wall_seconds': total_runtime,
        'files_per_second': len(results) / total_runtime,
        'mean_file_seconds': statistics.fmean(durations),
        'texture_score_mean': statistics.fmean(texture_scores),
        'texture_score_stddev': statistics.pstdev(texture_scores),
        'anomaly_score_mean': statistics.fmean(anomaly_scores),
        'outlier_score_mean': statistics.fmean(result['outlier_score'] for result in results),
        'mean_abs_spatial_corr': mean_abs_corr,
        'mean_high_freq_ratio': statistics.fmean(high_freq_ratio),
        'mean_hotspot_fraction': statistics.fmean(hotspot_fraction),
        'mean_entropy': statistics.fmean(entropy),
        'strongest_texture_file': {
            'file': strongest_texture['file'],
            'texture_score': strongest_texture['texture_score'],
        },
        'strongest_outlier_file': {
            'file': strongest_outlier['file'],
            'outlier_score': strongest_outlier['outlier_score'],
        },
        'slowest_file': {
            'file': slowest_file['file'],
            'duration_seconds': slowest_file['duration_seconds'],
        },
        'aggregate_mean': statistics.fmean(result['mean'] for result in results),
        'aggregate_std': statistics.fmean(result['std'] for result in results),
        'dataset_note': dataset_note,
        'worker_load': worker_load,
    }


def print_summary(summary):
    print('Analysis complete')
    mpi = summary.get('mpi')
    if mpi and mpi['enabled']:
        print(f"MPI ranks:           {mpi['size']}")
        print(f"Ranks with work:     {mpi['active_ranks']}")
        print(f"Files per rank:      min={mpi['min_files_per_rank']} max={mpi['max_files_per_rank']}")
    print(f"Files analyzed:      {summary['file_count']}")
    print(f"Workers used:        {summary['workers']}")
    print(f"Repeat count:        {summary['repeats']}")
    print(f"Actual runtime:      {summary['wall_seconds']:.2f} s")
    print(f"Throughput:          {summary['files_per_second']:.2f} files/s")
    print(f"Average file time:   {summary['mean_file_seconds']:.3f} s")
    print(f"Texture mean:        {summary['texture_score_mean']:.3f}")
    print(f"Texture std.dev:     {summary['texture_score_stddev']:.3f}")
    print(f"Anomaly mean:        {summary['anomaly_score_mean']:.3f}")
    print(f"Mean |corr|:         {summary['mean_abs_spatial_corr']:.5f}")
    print(f"Mean high-freq:      {summary['mean_high_freq_ratio']:.5f}")
    print(f"Mean hotspot frac:   {summary['mean_hotspot_fraction']:.5f}")
    print(f"Mean entropy:        {summary['mean_entropy']:.3f}")
    print(
        'Strongest texture:   '
        f"{summary['strongest_texture_file']['file']} ({summary['strongest_texture_file']['texture_score']:.3f})"
    )
    print(
        'Strongest outlier:   '
        f"{summary['strongest_outlier_file']['file']} ({summary['strongest_outlier_file']['outlier_score']:.3f})"
    )
    print(
        'Slowest file:        '
        f"{summary['slowest_file']['file']} ({summary['slowest_file']['duration_seconds']:.3f} s)"
    )
    print(f"Dataset note:        {summary['dataset_note']}")
    print('Worker file counts:')
    for worker in summary['worker_load']:
        print(f"  rank {worker['rank']} pid {worker['pid']}: {worker['files']} files")


def install_main_signal_handlers():
    previous_handlers = {}

    def handle_cancel(signum, _frame):
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f'Received {signame}')

    for signame in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        if hasattr(signal, signame):
            signum = getattr(signal, signame)
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_cancel)

    return previous_handlers


def restore_signal_handlers(previous_handlers):
    for signum, previous_handler in previous_handlers.items():
        signal.signal(signum, previous_handler)


def run_local_analysis(files, workers, repeats, chunk_size, progress_seconds, phase_prefix):
    if not files:
        log('[analysis] No files assigned to this rank.', prefix=phase_prefix)
        return [], 0.0

    context = get_context('spawn')
    pool = context.Pool(processes=workers, initializer=worker_initializer)
    try:
        start_time = time.perf_counter()
        log('[analysis] Starting full analysis run', prefix=phase_prefix)
        results = run_pool(
            pool,
            files,
            repeats=repeats,
            workers=workers,
            phase=f'{phase_prefix.strip()} analysis'.strip(),
            progress_seconds=progress_seconds,
            chunk_size=chunk_size,
        )
        total_runtime = time.perf_counter() - start_time
        pool.close()
        pool.join()
        return results, total_runtime
    except KeyboardInterrupt as exc:
        log(f'[cancel] {exc}. Terminating worker pool...', prefix=phase_prefix)
        pool.terminate()
        pool.join()
        log('[cancel] Worker pool terminated.', prefix=phase_prefix)
        raise
    except Exception:
        pool.terminate()
        pool.join()
        raise


def print_runtime_configuration(
    input_path,
    total_files,
    assigned_files,
    workers,
    repeats,
    chunk_size,
    mpi_context,
):
    phase_prefix = ''
    if mpi_context['enabled']:
        phase_prefix = f"[rank {mpi_context['rank']}/{mpi_context['size']}] "

    log('Runtime configuration', prefix=phase_prefix)
    log(f'Input path:           {input_path}', prefix=phase_prefix)
    log(f'Total files found:    {total_files}', prefix=phase_prefix)
    log(f'Files for this rank:  {assigned_files}', prefix=phase_prefix)
    log(f'CPU count detected:   {cpu_count() or 1}', prefix=phase_prefix)
    log(f'CPU budget:           {detected_cpu_budget()}', prefix=phase_prefix)
    log(f'Worker processes:     {workers}', prefix=phase_prefix)
    log(f'Repeat count:         {repeats}', prefix=phase_prefix)
    log(f'Chunk size:           {chunk_size}', prefix=phase_prefix)
    log('Start method:         spawn', prefix=phase_prefix)
    log(
        'Thread caps:         '
        f'OMP={os.environ.get("OMP_NUM_THREADS", "unset")}, '
        f'OPENBLAS={os.environ.get("OPENBLAS_NUM_THREADS", "unset")}, '
        f'MKL={os.environ.get("MKL_NUM_THREADS", "unset")}, '
        f'NUMEXPR={os.environ.get("NUMEXPR_NUM_THREADS", "unset")}',
        prefix=phase_prefix,
    )

    if mpi_context['enabled']:
        log(f"MPI rank:            {mpi_context['rank']} / {mpi_context['size']}", prefix=phase_prefix)
        log(f"MPI local rank:      {mpi_context['local_rank']}", prefix=phase_prefix)


def attach_mpi_summary(summary, mpi_context, files_per_rank, total_workers):
    summary['mpi'] = {
        'enabled': mpi_context['enabled'],
        'size': mpi_context['size'],
        'active_ranks': sum(1 for file_count in files_per_rank if file_count > 0),
        'min_files_per_rank': min(files_per_rank),
        'max_files_per_rank': max(files_per_rank),
        'total_worker_processes': total_workers,
    }


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f'Input path does not exist: {input_path}')

    files = sorted(input_path.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f'No .npy files found under: {input_path}')

    mpi_context = get_mpi_context()
    comm = mpi_context['comm']
    assigned_files = split_files_for_rank(files, mpi_context['rank'], mpi_context['size'])
    files_per_rank = [len(assigned_files)]
    if mpi_context['enabled']:
        files_per_rank = comm.allgather(len(assigned_files))

    workers = max(1, min(args.workers, len(assigned_files))) if assigned_files else 0
    repeats = max(1, args.repeats)
    chunk_size = args.chunk_files or max(1, len(assigned_files) // max(1, workers * 4))

    os.environ['ANALYSIS_MPI_RANK'] = str(mpi_context['rank'])
    os.environ['ANALYSIS_LOG_PREFIX'] = (
        f"[rank {mpi_context['rank']}/{mpi_context['size']}] " if mpi_context['enabled'] else ''
    )

    print_runtime_configuration(
        input_path=input_path,
        total_files=len(files),
        assigned_files=len(assigned_files),
        workers=workers,
        repeats=repeats,
        chunk_size=chunk_size,
        mpi_context=mpi_context,
    )

    signal_handlers = install_main_signal_handlers()
    try:
        if mpi_context['enabled']:
            comm.Barrier()

        phase_prefix = (
            f"[rank {mpi_context['rank']}/{mpi_context['size']}] " if mpi_context['enabled'] else ''
        )
        results, total_runtime = run_local_analysis(
            assigned_files,
            workers=workers,
            repeats=repeats,
            chunk_size=chunk_size,
            progress_seconds=args.progress_seconds,
            phase_prefix=phase_prefix,
        )
    except KeyboardInterrupt as exc:
        if mpi_context['enabled']:
            log(f'[cancel] {exc}. Aborting MPI job...', prefix=os.environ['ANALYSIS_LOG_PREFIX'])
            comm.Abort(130)
        return 130
    except Exception:
        if mpi_context['enabled']:
            log('[error] Unhandled exception, aborting MPI job.', prefix=os.environ['ANALYSIS_LOG_PREFIX'])
            comm.Abort(1)
        raise
    finally:
        restore_signal_handlers(signal_handlers)

    if mpi_context['enabled']:
        gathered_results = comm.gather(results, root=0)
        total_runtime = comm.reduce(total_runtime, op=MPI.MAX, root=0)
        total_workers = comm.reduce(workers, op=MPI.SUM, root=0)

        if mpi_context['rank'] != 0:
            return 0

        results = flatten(gathered_results)
        workers = total_workers

    summary = aggregate_results(
        results=results,
        total_runtime=total_runtime,
        workers=workers,
        repeats=repeats,
    )
    attach_mpi_summary(summary, mpi_context, files_per_rank, workers)
    print_summary(summary)

    if args.report_json is not None:
        report = {
            'summary': summary,
            'results': results,
        }
        args.report_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report written to:   {args.report_json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())