import { fireEvent, render } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";

import { Button } from "../src/components/ui/Button";

describe("Button", () => {
  it("renders children", () => {
    const { getByRole } = render(<Button>Save</Button>);
    expect(getByRole("button").textContent).toBe("Save");
  });

  it("defaults to type=button (not submit)", () => {
    const { getByRole } = render(<Button>Save</Button>);
    expect(getByRole("button").getAttribute("type")).toBe("button");
  });

  it("honors an explicit type=submit", () => {
    const { getByRole } = render(<Button type="submit">Send</Button>);
    expect(getByRole("button").getAttribute("type")).toBe("submit");
  });

  it("applies variant + size + tone classes", () => {
    const { getByRole } = render(
      <Button variant="secondary" size="sm" tone="danger">
        Delete
      </Button>,
    );
    const btn = getByRole("button");
    expect(btn.className).toContain("ui-button--secondary");
    expect(btn.className).toContain("ui-button--sm");
    expect(btn.className).toContain("ui-button--danger");
  });

  it("does not apply a tone class for neutral (the default)", () => {
    const { getByRole } = render(<Button>Save</Button>);
    expect(getByRole("button").className).not.toContain("ui-button--neutral");
  });

  it("loading renders an ellipsis and marks the button disabled", () => {
    const { getByRole } = render(<Button loading>Save</Button>);
    const btn = getByRole("button") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.textContent).toBe("…");
    expect(btn.dataset.loading).toBe("true");
  });

  it("disabled prevents onClick", () => {
    const onClick = vi.fn();
    const { getByRole } = render(
      <Button disabled onClick={onClick}>
        Save
      </Button>,
    );
    fireEvent.click(getByRole("button"));
    expect(onClick).not.toHaveBeenCalled();
  });

  it("fires onClick when enabled", () => {
    const onClick = vi.fn();
    const { getByRole } = render(<Button onClick={onClick}>Save</Button>);
    fireEvent.click(getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("forwards ref to the underlying button", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>Save</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("merges a consumer-supplied className", () => {
    const { getByRole } = render(<Button className="extra-class">Save</Button>);
    expect(getByRole("button").className).toContain("ui-button");
    expect(getByRole("button").className).toContain("extra-class");
  });

  it("icon variant defaults aria-label is unset (consumer must supply)", () => {
    const { getByRole } = render(
      <Button variant="icon" aria-label="Open menu" />,
    );
    expect(getByRole("button").getAttribute("aria-label")).toBe("Open menu");
  });

  it("passes through aria-pressed / aria-expanded", () => {
    const { getByRole } = render(
      <Button variant="icon" aria-label="Open menu" aria-expanded>
        ☰
      </Button>,
    );
    expect(getByRole("button").getAttribute("aria-expanded")).toBe("true");
  });
});
