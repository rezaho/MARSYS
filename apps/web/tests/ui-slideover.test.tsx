import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SlideOver } from "../src/components/ui/SlideOver";

describe("SlideOver", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <SlideOver open={false} onClose={() => undefined} ariaLabel="nav">
        body
      </SlideOver>,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders into a portal when open", () => {
    render(
      <SlideOver
        open={true}
        onClose={() => undefined}
        ariaLabel="nav"
        testId="panel"
      >
        <a href="#">link</a>
      </SlideOver>,
    );
    expect(document.body.querySelector("[data-testid='panel']")).not.toBeNull();
  });

  it("backdrop click invokes onClose", () => {
    const onClose = vi.fn();
    render(
      <SlideOver
        open={true}
        onClose={onClose}
        ariaLabel="nav"
        backdropTestId="bd"
      >
        <a href="#">link</a>
      </SlideOver>,
    );
    const bd = document.body.querySelector("[data-testid='bd']") as HTMLElement;
    fireEvent.mouseDown(bd);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("Escape closes the slide-over", () => {
    const onClose = vi.fn();
    render(
      <SlideOver open={true} onClose={onClose} ariaLabel="nav">
        <a href="#">link</a>
      </SlideOver>,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("default side is left", () => {
    render(
      <SlideOver
        open={true}
        onClose={() => undefined}
        ariaLabel="nav"
        testId="panel"
      >
        <a href="#">link</a>
      </SlideOver>,
    );
    const panel = document.body.querySelector(
      "[data-testid='panel']",
    ) as HTMLElement;
    expect(panel.className).toContain("ui-slide-over--left");
  });

  it("right side applies the right modifier class", () => {
    render(
      <SlideOver
        open={true}
        onClose={() => undefined}
        ariaLabel="nav"
        side="right"
        testId="panel"
      >
        <a href="#">link</a>
      </SlideOver>,
    );
    const panel = document.body.querySelector(
      "[data-testid='panel']",
    ) as HTMLElement;
    expect(panel.className).toContain("ui-slide-over--right");
  });

  it("width prop is applied via inline style", () => {
    render(
      <SlideOver
        open={true}
        onClose={() => undefined}
        ariaLabel="nav"
        width={320}
        testId="panel"
      >
        <a href="#">link</a>
      </SlideOver>,
    );
    const panel = document.body.querySelector(
      "[data-testid='panel']",
    ) as HTMLElement;
    expect(panel.style.width).toBe("320px");
  });
});
