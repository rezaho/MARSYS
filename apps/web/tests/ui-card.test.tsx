import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Card } from "../src/components/ui/Card";

describe("Card", () => {
  it("renders as a div by default", () => {
    const { container } = render(<Card>contents</Card>);
    expect(container.firstChild?.nodeName).toBe("DIV");
  });

  it("renders as a custom element when `as` is provided", () => {
    const { container } = render(<Card as="article">contents</Card>);
    expect(container.firstChild?.nodeName).toBe("ARTICLE");
  });

  it("renders as an anchor with href when `as=a`", () => {
    const { container } = render(
      <Card as="a" href="/foo">
        contents
      </Card>,
    );
    const a = container.firstChild as HTMLAnchorElement;
    expect(a.nodeName).toBe("A");
    expect(a.getAttribute("href")).toBe("/foo");
  });

  it("applies interactive class when prop set", () => {
    const { container } = render(<Card interactive>contents</Card>);
    expect((container.firstChild as HTMLElement).className).toContain(
      "ui-card--interactive",
    );
  });

  it("applies padding modifier class (default md)", () => {
    const { container, rerender } = render(<Card>contents</Card>);
    expect((container.firstChild as HTMLElement).className).toContain(
      "ui-card--padding-md",
    );
    rerender(<Card padding="sm">contents</Card>);
    expect((container.firstChild as HTMLElement).className).toContain(
      "ui-card--padding-sm",
    );
  });

  it("forwards a className prop", () => {
    const { container } = render(<Card className="my-card">contents</Card>);
    expect((container.firstChild as HTMLElement).className).toContain("my-card");
    expect((container.firstChild as HTMLElement).className).toContain("ui-card");
  });

  it("forwards arbitrary DOM props", () => {
    const { container } = render(
      <Card data-testid="my-card" aria-label="Test">
        contents
      </Card>,
    );
    expect((container.firstChild as HTMLElement).getAttribute("aria-label")).toBe(
      "Test",
    );
    expect((container.firstChild as HTMLElement).getAttribute("data-testid")).toBe(
      "my-card",
    );
  });
});
