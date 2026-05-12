import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { TagMarkup } from "../src/components/ui/TagMarkup";

describe("TagMarkup", () => {
  it("renders an inline tag with string attrs", () => {
    const { container } = render(
      <TagMarkup
        tag="agent"
        attrs={[
          ["name", "research"],
          ["model", "claude-opus-4-7"],
        ]}
      />,
    );
    expect(container.firstChild?.textContent).toBe(
      '<agent name="research" model="claude-opus-4-7" />',
    );
  });

  it("renders an empty tag when no attrs are supplied", () => {
    const { container } = render(<TagMarkup tag="workflow" attrs={[]} />);
    expect(container.firstChild?.textContent).toBe("<workflow />");
  });

  it("renders array attrs as brace-wrapped lists", () => {
    const { container } = render(
      <TagMarkup
        tag="agent"
        attrs={[["tools", ["search", "fetch"]]]}
      />,
    );
    expect(container.firstChild?.textContent).toBe("<agent tools={search,fetch} />");
  });

  it("block variant wraps in <pre> with line breaks per attr", () => {
    const { container } = render(
      <TagMarkup
        tag="agent"
        block
        attrs={[
          ["name", ""],
          ["model", ""],
          ["tools", []],
        ]}
      />,
    );
    expect(container.firstChild?.nodeName).toBe("PRE");
    const text = container.firstChild!.textContent!;
    // Indentation aligns the attrs after the opening tag.
    expect(text.startsWith("<agent name=\"\"")).toBe(true);
    expect(text).toContain("model=\"\"");
    expect(text).toContain("tools={}");
    expect(text.endsWith(" />")).toBe(true);
  });

  it("applies size modifier class", () => {
    const { container, rerender } = render(
      <TagMarkup tag="x" attrs={[]} size="xs" />,
    );
    expect((container.firstChild as HTMLElement).className).toContain(
      "ui-tagmarkup--xs",
    );
    rerender(<TagMarkup tag="x" attrs={[]} size="md" />);
    expect((container.firstChild as HTMLElement).className).toContain(
      "ui-tagmarkup--md",
    );
  });
});
