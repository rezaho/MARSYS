import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Kbd } from "../src/components/ui/Kbd";

describe("Kbd", () => {
  it("renders a <kbd> element with the design-system class", () => {
    const { container } = render(<Kbd>⌘</Kbd>);
    const k = container.firstChild as HTMLElement;
    expect(k.nodeName).toBe("KBD");
    expect(k.className).toContain("ui-kbd");
    expect(k.textContent).toBe("⌘");
  });

  it("merges a consumer className", () => {
    const { container } = render(<Kbd className="extra">K</Kbd>);
    expect((container.firstChild as HTMLElement).className).toContain("ui-kbd");
    expect((container.firstChild as HTMLElement).className).toContain("extra");
  });
});
