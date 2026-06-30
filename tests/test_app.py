import unittest

import app


class ProvenanceGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.app.test_client()

    def test_submit_returns_label_and_confidence(self) -> None:
        response = self.client.post(
            "/submit",
            json={
                "text": "Artificial intelligence represents a transformative paradigm shift in modern society.",
                "creator_id": "tester",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("content_id", payload)
        self.assertIn("confidence", payload)
        self.assertIn("label", payload)
        self.assertIn("transparency_label", payload)

    def test_appeal_updates_status_and_log(self) -> None:
        submit_response = self.client.post(
            "/submit",
            json={
                "text": "This is a test submission for the appeal workflow.",
                "creator_id": "tester",
            },
        )
        content_id = submit_response.get_json()["content_id"]
        appeal_response = self.client.post(
            "/appeal",
            json={
                "content_id": content_id,
                "creator_reasoning": "I believe the system misclassified this text.",
            },
        )
        self.assertEqual(appeal_response.status_code, 200)
        self.assertEqual(appeal_response.get_json()["status"], "under_review")

        log_response = self.client.get("/log")
        payload = log_response.get_json()
        self.assertTrue(payload["entries"])
        self.assertTrue(any(entry["event_type"] == "appeal" for entry in payload["entries"]))

    def test_analytics_endpoint_reports_metrics(self) -> None:
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("total_submissions", payload)
        self.assertIn("detection_pattern", payload)
        self.assertIn("appeal_rate", payload)
        self.assertIn("average_confidence", payload)


if __name__ == "__main__":
    unittest.main()
