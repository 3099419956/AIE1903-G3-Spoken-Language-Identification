import unittest

from scripts.http_form import parse_multipart_form, safe_wav_name


class MultipartFormTests(unittest.TestCase):
    def test_parses_audio_and_source(self) -> None:
        boundary = "----aie1903-test"
        audio = b"RIFF-test-audio"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="source"\r\n\r\n'
            "your recording\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="clip.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode("utf-8")

        raw, filename, fields = parse_multipart_form(
            f"multipart/form-data; boundary={boundary}", body
        )

        self.assertEqual(raw, audio)
        self.assertEqual(filename, "clip.wav")
        self.assertEqual(fields, {"source": "your recording"})

    def test_rejects_non_multipart_request(self) -> None:
        with self.assertRaises(ValueError):
            parse_multipart_form("application/json", b"{}")

    def test_upload_filename_cannot_escape_recordings_folder(self) -> None:
        self.assertEqual(safe_wav_name(r"..\..\private.mp3"), "private.mp3.wav")


if __name__ == "__main__":
    unittest.main()
