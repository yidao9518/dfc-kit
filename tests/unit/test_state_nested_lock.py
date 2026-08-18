import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.io.state_nested_lock import (
    NestedCheckpointBusyError,
    acquire_nested_checkpoint_lock,
    inspect_nested_checkpoint_lock,
    nested_checkpoint_lock_path,
)


class NestedCheckpointLockTests(unittest.TestCase):
    def test_active_owner_is_visible_and_second_owner_is_rejected(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "nested.checkpoint"
            with acquire_nested_checkpoint_lock(checkpoint) as lock_path:
                observed = inspect_nested_checkpoint_lock(checkpoint)
                self.assertEqual(lock_path, nested_checkpoint_lock_path(checkpoint))
                self.assertEqual(observed.status, "active")
                self.assertEqual(observed.pid, os.getpid())
                self.assertTrue(observed.hostname)
                self.assertIsNotNone(observed.acquired_at_unix)
                self.assertIsNone(observed.released_at_unix)

                before = lock_path.read_bytes()
                with self.assertRaisesRegex(
                    NestedCheckpointBusyError,
                    rf"pid {os.getpid()}.*already active|already active.*pid {os.getpid()}",
                ), acquire_nested_checkpoint_lock(checkpoint):
                    self.fail("a second owner acquired the same checkpoint")
                self.assertEqual(lock_path.read_bytes(), before)

            released = inspect_nested_checkpoint_lock(checkpoint)
            self.assertEqual(released.status, "idle")
            self.assertEqual(released.pid, os.getpid())
            self.assertIsNotNone(released.released_at_unix)
            self.assertGreaterEqual(
                released.released_at_unix,
                released.acquired_at_unix,
            )

    def test_other_process_owns_lock_until_normal_release(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "nested.checkpoint"
            script = """
import sys
from dfckit.io.state_nested_lock import acquire_nested_checkpoint_lock

with acquire_nested_checkpoint_lock(sys.argv[1]):
    print("locked", flush=True)
    sys.stdin.readline()
"""
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(checkpoint)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(process.stdout.readline(), "locked\n")
                active = inspect_nested_checkpoint_lock(checkpoint)
                self.assertEqual(active.status, "active")
                self.assertEqual(active.pid, process.pid)
                with self.assertRaises(
                    NestedCheckpointBusyError
                ), acquire_nested_checkpoint_lock(checkpoint):
                    self.fail("the parent acquired a child-owned checkpoint")
                stdout, stderr = process.communicate("release\n", timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout, "")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
            released = inspect_nested_checkpoint_lock(checkpoint)
            self.assertEqual(released.status, "idle")
            self.assertEqual(released.pid, process.pid)

    @unittest.skipUnless(os.name == "posix", "hard-exit recovery uses POSIX flock")
    def test_hard_exit_is_stale_and_safely_reclaimed(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "nested.checkpoint"
            script = """
import os
import sys
from dfckit.io.state_nested_lock import acquire_nested_checkpoint_lock

with acquire_nested_checkpoint_lock(sys.argv[1]):
    print("locked", flush=True)
    os._exit(17)
"""
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(checkpoint)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 17, stderr)
            self.assertEqual(stdout, "locked\n")

            stale = inspect_nested_checkpoint_lock(checkpoint)
            self.assertEqual(stale.status, "stale")
            self.assertEqual(stale.pid, process.pid)
            self.assertIsNone(stale.released_at_unix)

            with acquire_nested_checkpoint_lock(checkpoint):
                reclaimed = inspect_nested_checkpoint_lock(checkpoint)
                self.assertEqual(reclaimed.status, "active")
                self.assertEqual(reclaimed.pid, os.getpid())
            self.assertEqual(inspect_nested_checkpoint_lock(checkpoint).status, "idle")

    def test_missing_malformed_and_unsafe_lock_paths_are_distinguished(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "nested.checkpoint"
            lock_path = nested_checkpoint_lock_path(checkpoint)
            missing = inspect_nested_checkpoint_lock(checkpoint)
            self.assertEqual(missing.status, "idle")
            self.assertIsNone(missing.pid)

            lock_path.write_text("not JSON", encoding="utf-8")
            invalid = inspect_nested_checkpoint_lock(checkpoint)
            self.assertEqual(invalid.status, "invalid")
            self.assertIsNone(invalid.pid)

            lock_path.unlink()
            target = Path(temporary) / "outside.json"
            target.write_text(json.dumps({"preserve": True}), encoding="utf-8")
            lock_path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                inspect_nested_checkpoint_lock(checkpoint)
            with self.assertRaisesRegex(
                ValueError,
                "must not be a symlink",
            ), acquire_nested_checkpoint_lock(checkpoint):
                self.fail("a symlinked lock path was accepted")
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"preserve": True})

            lock_path.unlink()
            lock_path.mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                inspect_nested_checkpoint_lock(checkpoint)
            with self.assertRaisesRegex(
                ValueError,
                "regular file",
            ), acquire_nested_checkpoint_lock(checkpoint):
                self.fail("a directory lock path was accepted")


if __name__ == "__main__":
    unittest.main()
