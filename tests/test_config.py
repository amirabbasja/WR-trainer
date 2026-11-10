import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from workrunner.config import ConfigError, DEFAULT_CONFIG, load_config, read_config, save_config, update_config
from workrunner.config_export import _encrypt_object


class ConfigSchemeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "config.json"
        self.values = copy.deepcopy(DEFAULT_CONFIG)
        self.values["vpn"]["link"] = "vless://fixture@example.invalid:443"
        self.values["S3"]["WR_S2"] = "PRIVATE-IDENTITY"
        self.values["S3"]["password"] = "PRIVATE-S3-PASSWORD"
        self.values["training"]["output_dir"] = "results"
        self.values["side_job"]["repo_url"] = "https://huggingface.co/datasets/fixture/thesis"

    def test_legacy_node_address_is_dropped_when_loading_and_saving(self):
        self.values["node"]["address"] = "legacy-value"
        self.path.write_text(json.dumps(self.values))

        self.assertNotIn("address", read_config(self.path)["node"])
        self.assertNotIn("address", load_config(self.path)["node"])
        save_config(self.path, self.values)
        self.assertNotIn("address", json.loads(self.path.read_text())["node"])

    def test_only_s3_encrypted_and_no_env_or_key_file_created(self):
        original = copy.deepcopy(self.values)
        save_config(self.path, self.values)
        stored = json.loads(self.path.read_text())
        self.assertNotIn("S3", stored)
        self.assertIn("ciphertext", stored["S_3"])
        for name in original.keys() - {"S3"}:
            self.assertEqual(stored[name], original[name])
        self.assertNotIn("PRIVATE-IDENTITY", self.path.read_text())
        self.assertNotIn("PRIVATE-S3-PASSWORD", self.path.read_text())
        self.assertEqual(read_config(self.path), original)
        self.assertEqual(read_config(self.path, key="qwe"), original)
        self.assertEqual(self.values, original)
        self.assertEqual(set(self.directory.iterdir()), {self.path})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_plain_settings_can_be_edited_without_reencrypting_s3(self):
        save_config(self.path, self.values)
        stored = json.loads(self.path.read_text())
        encrypted = copy.deepcopy(stored["S_3"])
        stored["training"]["max_batches"] = 7
        self.path.write_text(json.dumps(stored))
        loaded = load_config(self.path)
        self.assertEqual(loaded["training"]["max_batches"], 7)
        self.assertEqual(loaded["S3"]["WR_S2"], "PRIVATE-IDENTITY")
        self.assertEqual(json.loads(self.path.read_text())["S_3"], encrypted)
        self.assertEqual(loaded["training"]["output_dir"], str(self.directory / "results"))

    def test_nested_updates_preserve_plain_siblings_and_encrypt_s3(self):
        save_config(self.path, self.values)
        update_config(self.path, {"S3": {"worker": "updated"}, "training": {"max_batches": 3}})
        expected = copy.deepcopy(self.values)
        expected["S3"]["worker"] = "updated"
        expected["training"]["max_batches"] = 3
        self.assertEqual(read_config(self.path), expected)
        self.assertNotIn("S3", json.loads(self.path.read_text()))

    def test_custom_key_requires_explicit_override_and_updates_round_trip(self):
        save_config(self.path, self.values, key="custom key")
        before = self.path.read_bytes()
        with self.assertRaises(ConfigError):
            read_config(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        update_config(self.path, {"training": {"max_batches": 2}}, key="custom key")
        self.assertEqual(read_config(self.path, key="custom key")["training"]["max_batches"], 2)

    def test_dotenv_and_process_environment_are_not_read(self):
        (self.directory / ".env").write_text("open_text=wrong\n")
        with patch.dict(os.environ, {"open_text": "wrong", "LRAEP_CONFIG_KEY": "wrong"}):
            save_config(self.path, self.values)
            self.assertEqual(read_config(self.path), self.values)
        self.assertEqual((self.directory / ".env").read_text(), "open_text=wrong\n")

    def test_plain_and_old_whole_encrypted_documents_migrate_without_losing_values(self):
        for document, key in ((self.values, None), (_encrypt_object(self.values, "old key"), "old key")):
            self.path.write_text(json.dumps(document))
            self.assertEqual(read_config(self.path, key=key), self.values)
            save_config(self.path, read_config(self.path, key=key))
            self.assertEqual(read_config(self.path), self.values)
            self.assertIn("S_3", json.loads(self.path.read_text()))

    def test_invalid_write_preserves_existing_file_and_cleans_temporary_files(self):
        save_config(self.path, self.values)
        before = self.path.read_bytes()
        for changes in ({"S3": {"port": 0}}, {"training": {"max_batches": -1}}):
            with self.assertRaises(ConfigError):
                update_config(self.path, changes)
            self.assertEqual(self.path.read_bytes(), before)
        with patch("workrunner.config.os.replace", side_effect=OSError("PRIVATE-ERROR")):
            with self.assertRaises(ConfigError) as caught:
                update_config(self.path, {"training": {"max_batches": 5}})
        self.assertNotIn("PRIVATE-ERROR", str(caught.exception))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_ambiguous_section_names_and_tampering_fail_without_secret_values(self):
        save_config(self.path, self.values)
        original = json.loads(self.path.read_text())
        for document in (dict(original, S3={}), dict(original, S_3={"ciphertext": "PRIVATE"})):
            self.path.write_text(json.dumps(document))
            with self.assertRaises(ConfigError) as caught:
                load_config(self.path)
            self.assertNotIn("PRIVATE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
