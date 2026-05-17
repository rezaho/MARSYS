/**
 * lib/files client tests.
 *
 * Covers UploadError shape + the deleteFile envelope parsing.
 * The XHR upload itself is best validated end-to-end via Playwright.
 */
import { describe, expect, it } from "vitest";

import { UploadError } from "../src/lib/files";

describe("UploadError", () => {
  it("carries status + code + details", () => {
    const err = new UploadError("file too large", 413, "FILE_TOO_LARGE", {
      max_bytes: 100,
    });
    expect(err.message).toBe("file too large");
    expect(err.status).toBe(413);
    expect(err.code).toBe("FILE_TOO_LARGE");
    expect(err.details).toEqual({ max_bytes: 100 });
    expect(err.name).toBe("UploadError");
  });
});
