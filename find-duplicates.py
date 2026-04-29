import argparse
import os
import sys
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(
		description='Compare .npy matrices in threads and find inverse-matrix pairs.'
	)
	parser.add_argument('input_path', nargs='?', default='super_data')
	parser.add_argument(
		'--recursive',
		action='store_true',
		help='Search for .npy files recursively.',
	)
	parser.add_argument(
		'--rtol',
		type=float,
		default=0.1,
		help='Relative tolerance for the inverse matrix check.',
	)
	parser.add_argument(
		'--atol',
		type=float,
		default=0.1,
		help='Absolute tolerance for the inverse matrix check.',
	)
	parser.add_argument(
		'--max-threads',
		type=int,
		default=max(1, (os.cpu_count() or 1) - 1),
		help='Maximum number of worker threads to run at the same time.',
	)
	parser.add_argument(
		'--limit',
		type=int,
		default=5,
		help='Only compare the first N files. Use 0 to compare all files.',
	)
	return parser.parse_args()


def collect_npy_files(input_path, recursive=False):
	pattern = '**/*.npy' if recursive else '*.npy'
	files = sorted(input_path.glob(pattern))
	return [file_path for file_path in files if file_path.is_file()]


def limit_file_paths(file_paths, limit):
	if limit <= 0:
		return file_paths
	return file_paths[:limit]


def load_array(file_path, array_cache, cache_lock):
	with cache_lock:
		cached_array = array_cache.get(file_path)
		if cached_array is not None:
			return cached_array

	loaded_array = np.load(file_path, allow_pickle=False)

	with cache_lock:
		array_cache[file_path] = loaded_array

	return loaded_array


def can_compare_as_inverse(left_array, right_array):
	if left_array.ndim != 2 or right_array.ndim != 2:
		return False
	if left_array.shape[0] != left_array.shape[1]:
		return False
	return left_array.shape == right_array.shape


def invert_array(array):
	if array.ndim != 2 or array.shape[0] != array.shape[1]:
		return None

	try:
		return np.linalg.inv(array)
	except np.linalg.LinAlgError:
		return None


def inverse_matches_array(inverse_array, other_array, rtol, atol):
	if inverse_array is None:
		return False
	if other_array.ndim != 2:
		return False
	if inverse_array.shape != other_array.shape:
		return False
	return np.allclose(inverse_array, other_array, rtol=rtol, atol=atol)


def files_form_pair(left_path, right_path, array_cache, cache_lock, rtol, atol):
	left_array = load_array(left_path, array_cache, cache_lock)
	right_array = load_array(right_path, array_cache, cache_lock)
	if not can_compare_as_inverse(left_array, right_array):
		return False

	left_inverse = invert_array(left_array)
	return inverse_matches_array(left_inverse, right_array, rtol=rtol, atol=atol)


def compare_file_to_remaining(
	file_index,
	file_paths,
	array_cache,
	cache_lock,
	rtol,
	atol,
	progress_callback=None,
):
	source_path = file_paths[file_index]
	source_array = load_array(source_path, array_cache, cache_lock)
	source_inverse = invert_array(source_array)
	local_matches = []

	for other_index in range(file_index + 1, len(file_paths)):
		other_path = file_paths[other_index]
		other_array = load_array(other_path, array_cache, cache_lock)
		if inverse_matches_array(source_inverse, other_array, rtol, atol):
			local_matches.append((source_path, other_path))
		if progress_callback is not None:
			progress_callback(other_path, other_index - file_index)

	return local_matches


def create_comparison_jobs(file_paths):
	return list(range(max(0, len(file_paths) - 1)))


def print_message(message, output_lock):
	with output_lock:
		print(message, flush=True)


def worker_loop(
	job_queue,
	file_paths,
	pair_matches,
	pair_lock,
	progress,
	progress_lock,
	output_lock,
	array_cache,
	cache_lock,
	rtol,
	atol,
):
	while True:
		file_index = job_queue.get()
		if file_index is None:
			job_queue.task_done()
			return

		source_path = file_paths[file_index]
		remaining_comparisons = len(file_paths) - file_index - 1
		print_message(
			f'[start] {source_path.name}: comparing against {remaining_comparisons} files',
			output_lock,
		)

		local_matches = []

		def on_comparison(other_path, completed_comparisons):
			print_message(
				f'[file] {source_path.name}: {completed_comparisons}/{remaining_comparisons} '
				f'checked against {other_path.name}',
				output_lock,
			)

		local_matches = compare_file_to_remaining(
			file_index,
			file_paths,
			array_cache,
			cache_lock,
			rtol,
			atol,
			progress_callback=on_comparison,
		)

		if local_matches:
			with pair_lock:
				pair_matches.extend(local_matches)
				matches_found = len(pair_matches)
		else:
			with pair_lock:
				matches_found = len(pair_matches)

		with progress_lock:
			progress['completed_jobs'] += 1
			completed_jobs = progress['completed_jobs']
			total_jobs = progress['total_jobs']

		print_message(
			f'[{completed_jobs}/{total_jobs}] compared {file_paths[file_index].name} -> '
			f'{len(local_matches)} matches, total matches={matches_found}',
			output_lock,
		)
		job_queue.task_done()


def start_worker_threads(
	worker_count,
	job_queue,
	file_paths,
	pair_matches,
	pair_lock,
	progress,
	progress_lock,
	output_lock,
	array_cache,
	cache_lock,
	rtol,
	atol,
):
	threads = []
	for worker_index in range(worker_count):
		thread = Thread(
			name=f'comparison-worker-{worker_index + 1}',
			target=worker_loop,
			args=(
				job_queue,
				file_paths,
				pair_matches,
				pair_lock,
				progress,
				progress_lock,
				output_lock,
				array_cache,
				cache_lock,
				rtol,
				atol,
			),
		)
		thread.start()
		threads.append(thread)
	return threads


def enqueue_jobs(job_queue, jobs):
	for job in jobs:
		job_queue.put(job)


def stop_worker_threads(job_queue, worker_count):
	for _ in range(worker_count):
		job_queue.put(None)


def wait_for_threads(threads):
	for thread in threads:
		thread.join()


def print_startup_summary(file_paths, worker_count, total_jobs, rtol, atol):
	print(f'Files scanned: {len(file_paths)}')
	print(f'Comparison jobs: {total_jobs}')
	print(f'Worker threads: {worker_count}')
	print(f'Inverse tolerance: rtol={rtol} atol={atol}')
	print('Starting comparisons...', flush=True)


def print_pair_matches(file_paths, pair_matches):
	print(f'Files scanned: {len(file_paths)}')
	print(f'Pairs found: {len(pair_matches)}')
	for left_path, right_path in pair_matches:
		print(f'{left_path} inverse -> {right_path}')


def main():
	args = parse_args()
	input_path = Path(args.input_path)

	file_paths = collect_npy_files(input_path, recursive=args.recursive)
	file_paths = limit_file_paths(file_paths, args.limit)
	if not file_paths:
		raise FileNotFoundError(f'No .npy files found under: {input_path}')

	jobs = create_comparison_jobs(file_paths)
	worker_count = min(max(1, args.max_threads), max(1, len(jobs)))
	pair_matches = []
	pair_lock = Lock()
	progress = {'completed_jobs': 0, 'total_jobs': len(jobs)}
	progress_lock = Lock()
	output_lock = Lock()
	array_cache = {}
	cache_lock = Lock()
	job_queue = Queue()

	print_startup_summary(file_paths, worker_count, len(jobs), args.rtol, args.atol)
	threads = start_worker_threads(
		worker_count,
		job_queue,
		file_paths,
		pair_matches,
		pair_lock,
		progress,
		progress_lock,
		output_lock,
		array_cache,
		cache_lock,
		args.rtol,
		args.atol,
	)
	enqueue_jobs(job_queue, jobs)
	job_queue.join()
	stop_worker_threads(job_queue, worker_count)

	wait_for_threads(threads)

	print_pair_matches(file_paths, pair_matches)
	return 0


if __name__ == '__main__':
	sys.exit(main())

