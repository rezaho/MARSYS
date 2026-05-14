"""File upload + lookup module.

Owns the multipart-streaming upload handler (capped per-file + aggregate),
the lookup helper consumed by both the download endpoint + the run
lifecycle's attachment-resolution step, and the on-disk path layout
(``<data-dir>/data/files/{file_id}/<original_filename>``).
"""
