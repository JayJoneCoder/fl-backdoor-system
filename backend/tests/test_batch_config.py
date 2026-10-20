"""Batch uploads validate structure without writing config or starting workers."""

import io
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from backend import config_manager, main


class BatchConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def parse(self, payload):
        return config_manager.parse_batch_experiments(json.dumps(payload).encode("utf-8"))

    def test_common_config_is_merged_without_mutation(self):
        payload = {"common": {"attack": "none", "dataset": "mnist"}, "experiments": [
            {"name": "first"}, {"name": "second", "attack": "badnets"},
        ]}
        self.assertEqual(self.parse(payload), [
            {"name": "first", "attack": "none", "dataset": "mnist"},
            {"name": "second", "attack": "badnets", "dataset": "mnist"},
        ])
        self.assertEqual(payload["experiments"][0], {"name": "first"})
        self.assertEqual(self.parse([{"name": "数组格式"}]), [{"name": "数组格式"}])

    def test_invalid_structure_has_readable_item_or_field_error(self):
        for payload, message in (
            ([None], "第 1 项实验必须"), ([1], "第 1 项实验必须"),
            ([{}], "第 1 项实验：name"), ([{"name": 1}], "第 1 项实验：name"),
            ([{"name": " "}], "第 1 项实验：name"),
            ([{"name": "one"}, {"name": "ONE"}], "第 2 项实验：名称"),
            ([{"name": "../bad"}], "第 1 项实验：名称不合法"),
            ([], "非空数组"), ({"experiments": {}}, "非空数组"),
            ({"experiments": [{"name": "one"}], "common": None}, "common 必须"),
            ({"experiments": [{"name": "one"}], "common": []}, "common 必须"),
            ({}, "批量配置必须"), (None, "批量配置必须"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                self.parse(payload)

    async def test_parse_endpoint_returns_400_without_creating_tasks_or_running(self):
        with patch.object(main.batch_manager, "create_task") as create, patch.object(main.runner, "start") as start:
            for content, message in ((b'[null]', "第 1 项实验"), (b'[{', "JSON 格式错误"), (b'\xff', "编码无效")):
                with self.subTest(content=content), self.assertRaises(HTTPException) as caught:
                    await main.parse_batch_config(UploadFile(file=io.BytesIO(content), filename="bad.json"))
                self.assertEqual(caught.exception.status_code, 400)
                self.assertIn(message, caught.exception.detail)
            create.assert_not_called()
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()