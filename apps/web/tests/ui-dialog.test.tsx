import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Dialog } from "../src/components/ui/Dialog";

describe("Dialog", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <Dialog open={false} onClose={() => undefined} ariaLabel="x">
        body
      </Dialog>,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders into a portal on document.body", () => {
    render(
      <Dialog
        open={true}
        onClose={() => undefined}
        ariaLabel="Pick something"
        testId="dlg"
      >
        <button>inside</button>
      </Dialog>,
    );
    const dlg = document.body.querySelector("[data-testid='dlg']");
    expect(dlg).not.toBeNull();
  });

  it("backdrop click invokes onClose", () => {
    const onClose = vi.fn();
    render(
      <Dialog
        open={true}
        onClose={onClose}
        ariaLabel="Pick something"
        testId="dlg"
      >
        <button>inside</button>
      </Dialog>,
    );
    const backdrop = document.body.querySelector(
      "[data-testid='dlg']",
    ) as HTMLElement;
    fireEvent.mouseDown(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("clicks inside the dialog body do NOT close", () => {
    const onClose = vi.fn();
    render(
      <Dialog
        open={true}
        onClose={onClose}
        ariaLabel="Pick something"
        testId="dlg"
      >
        <button data-testid="inner-btn">inside</button>
      </Dialog>,
    );
    const inner = document.body.querySelector(
      "[data-testid='inner-btn']",
    ) as HTMLElement;
    fireEvent.mouseDown(inner);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("Escape closes the dialog", () => {
    const onClose = vi.fn();
    render(
      <Dialog open={true} onClose={onClose} ariaLabel="dlg">
        <button>x</button>
      </Dialog>,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("aria-modal + aria-label set correctly", () => {
    render(
      <Dialog open={true} onClose={() => undefined} ariaLabel="My label" testId="dlg">
        <button>x</button>
      </Dialog>,
    );
    const dialog = document.body.querySelector("[role='dialog']") as HTMLElement;
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.getAttribute("aria-label")).toBe("My label");
  });

  it("position class reflects the position prop", () => {
    render(
      <Dialog
        open={true}
        onClose={() => undefined}
        ariaLabel="x"
        position="bottom"
        testId="dlg"
      >
        <button>x</button>
      </Dialog>,
    );
    const backdrop = document.body.querySelector("[data-testid='dlg']") as HTMLElement;
    expect(backdrop.className).toContain("ui-dialog-backdrop--bottom");
    const dialog = document.body.querySelector("[role='dialog']") as HTMLElement;
    expect(dialog.className).toContain("ui-dialog--bottom");
  });

  it("locks body scroll while open", () => {
    document.body.style.overflow = "";
    const { rerender } = render(
      <Dialog open={true} onClose={() => undefined} ariaLabel="x">
        <button>x</button>
      </Dialog>,
    );
    expect(document.body.style.overflow).toBe("hidden");
    rerender(
      <Dialog open={false} onClose={() => undefined} ariaLabel="x">
        <button>x</button>
      </Dialog>,
    );
    expect(document.body.style.overflow).toBe("");
  });
});
