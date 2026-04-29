import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np


def load_find_duplicates_module():
    module_path = Path(__file__).resolve().parents[1] / 'find-duplicates.py'
    spec = importlib.util.spec_from_file_location('find_duplicates_module', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


find_duplicates = load_find_duplicates_module()


class FileComparisonTests(unittest.TestCase):
    def setUp(self):
        self.array_cache = {}
        self.cache_lock = Lock()

    def save_array(self, directory, file_name, array):
        file_path = Path(directory) / file_name
        np.save(file_path, array)
        return file_path

    def test_files_form_pair_when_right_file_is_inverse(self):
        with TemporaryDirectory() as temp_dir:
            left_path = self.save_array(
                temp_dir,
                'left.npy',
                np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64),
            )
            right_path = self.save_array(
                temp_dir,
                'right.npy',
                np.array([[0.5, 0.0], [0.0, 2.0]], dtype=np.float64),
            )

            result = find_duplicates.files_form_pair(
                left_path,
                right_path,
                self.array_cache,
                self.cache_lock,
                rtol=1e-8,
                atol=1e-8,
            )

            self.assertTrue(result)

    def test_files_form_pair_returns_false_when_right_file_is_not_inverse(self):
        with TemporaryDirectory() as temp_dir:
            left_path = self.save_array(
                temp_dir,
                'left.npy',
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )
            right_path = self.save_array(
                temp_dir,
                'right.npy',
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            )

            result = find_duplicates.files_form_pair(
                left_path,
                right_path,
                self.array_cache,
                self.cache_lock,
                rtol=1e-8,
                atol=1e-8,
            )

            self.assertFalse(result)

    def test_files_form_pair_returns_false_for_singular_matrix(self):
        with TemporaryDirectory() as temp_dir:
            left_path = self.save_array(
                temp_dir,
                'left.npy',
                np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64),
            )
            right_path = self.save_array(
                temp_dir,
                'right.npy',
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            )

            result = find_duplicates.files_form_pair(
                left_path,
                right_path,
                self.array_cache,
                self.cache_lock,
                rtol=1e-8,
                atol=1e-8,
            )

            self.assertFalse(result)

    def test_compare_file_to_remaining_only_returns_forward_matches(self):
        with TemporaryDirectory() as temp_dir:
            source_path = self.save_array(
                temp_dir,
                'source.npy',
                np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64),
            )
            matching_path = self.save_array(
                temp_dir,
                'matching.npy',
                np.array([[0.5, 0.0], [0.0, 2.0]], dtype=np.float64),
            )
            non_matching_path = self.save_array(
                temp_dir,
                'other.npy',
                np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float64),
            )

            file_paths = [source_path, matching_path, non_matching_path]
            result = find_duplicates.compare_file_to_remaining(
                0,
                file_paths,
                self.array_cache,
                self.cache_lock,
                rtol=1e-8,
                atol=1e-8,
            )

            self.assertEqual(result, [(source_path, matching_path)])


if __name__ == '__main__':
    unittest.main()